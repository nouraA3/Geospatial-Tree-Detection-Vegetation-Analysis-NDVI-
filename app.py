import sys
import cv2
import numpy as np
import onnxruntime as ort
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import rasterio

# ---- Model metrics ----
MODEL_MAP50 = 0.8776  # 87.76%[cite: 7]


class ZoomableGraphicsView(QGraphicsView):
  """Graphics view with zooming and panning support."""

  def __init__(self):
    super().__init__()
    self.scale_factor = 1.1
    self.setDragMode(QGraphicsView.ScrollHandDrag)

  def wheelEvent(self, event):
    # Hold CTRL to zoom with mouse wheel
    if QApplication.keyboardModifiers() == Qt.ControlModifier:
      if event.angleDelta().y() > 0:
        self.scale(self.scale_factor, self.scale_factor)
      else:
        self.scale(1 / self.scale_factor, 1 / self.scale_factor)
    else:
      super().wheelEvent(event)


class ImageViewer(QWidget):
  """Main window: RGB / NDVI viewer + tree counting."""

  def __init__(self, tif_path: str = "", onnx_model_path: str = "tree_model.onnx"):
    super().__init__()

    self.setWindowTitle("Tree Detection & NDVI Analysis Viewer")
    self.setGeometry(100, 100, 1000, 850)

    self.tif_path = tif_path
    self.onnx_model_path = onnx_model_path
    self.rgb_image = QImage()

    # Main vertical layout
    self.layout = QVBoxLayout()
    self.setLayout(self.layout)

    # Image viewer (with zoom & pan)
    self.viewer = ZoomableGraphicsView()
    self.viewer.setRenderHint(QPainter.Antialiasing)
    self.layout.addWidget(self.viewer)

    # ---------- Upload Button ----------
    self.upload_button = QPushButton("📂 Upload GeoTIFF Image (.tif)")
    self.upload_button.setStyleSheet(
        "font-size: 14px; font-weight: bold; background-color: #2b5c8f; color:"
        " white; padding: 6px;"
    )
    self.upload_button.clicked.connect(self.upload_image)
    self.layout.addWidget(self.upload_button)

    # ---------- Processing Buttons ----------
    self.rgb_button = QPushButton("Show RGB")
    self.rgb_button.clicked.connect(self.show_rgb)

    self.ndvi_raw_button = QPushButton("Show NDVI (grayscale)")
    self.ndvi_raw_button.clicked.connect(self.show_ndvi_raw)

    self.ndvi_heatmap_button = QPushButton(
        "Show NDVI Heatmap (Viridis colormap)"
    )
    self.ndvi_heatmap_button.clicked.connect(self.show_ndvi_heatmap)

    self.tree_button = QPushButton("Count Trees")
    self.tree_button.clicked.connect(self.count_trees)

    self.layout.addWidget(self.rgb_button)
    self.layout.addWidget(self.ndvi_raw_button)
    self.layout.addWidget(self.ndvi_heatmap_button)
    self.layout.addWidget(self.tree_button)

    # ---------- Info labels ----------
    # File Path
    self.path_label = QLabel(
        f"Selected File: {self.tif_path if self.tif_path else 'No file uploaded'}"
    )
    self.path_label.setStyleSheet("font-size: 13px; color: #555555;")
    self.layout.addWidget(self.path_label)

    # Tree count
    self.count_label = QLabel("Trees: 0")
    self.count_label.setStyleSheet(
        "font-size: 17px; font-weight: bold; color: green;"
    )
    self.layout.addWidget(self.count_label)

    # Model accuracy
    self.accuracy_label = QLabel(
        "Tree Counting Model Accuracy (mAP50):"
        f" {MODEL_MAP50 * 100:.2f}%"  #[cite: 7]
    )
    self.accuracy_label.setStyleSheet("font-size: 14px; color: #333333;")
    self.layout.addWidget(self.accuracy_label)

    # NDVI legend
    self.legend_label = QLabel("")
    self.legend_label.setStyleSheet("font-size: 13px; color: #333333;")
    self.layout.addWidget(self.legend_label)

    # ---------- Load ONNX model ----------
    try:
      self.ort_session = ort.InferenceSession(
          self.onnx_model_path,
          providers=["CPUExecutionProvider"],
      )
      self.ort_input_name = self.ort_session.get_inputs()[0].name
      in_shape = self.ort_session.get_inputs()[0].shape  # [1, 3, H, W]
      self.model_input_height = (
          int(in_shape[2]) if isinstance(in_shape[2], int) else 640
      )
      self.model_input_width = (
          int(in_shape[3]) if isinstance(in_shape[3], int) else 640
      )
    except Exception as e:
      self.count_label.setText("Error loading model!")
      print(f"Error loading ONNX model: {e}")
      self.model_input_height = 640
      self.model_input_width = 640

    # Load initial image if path provided
    if self.tif_path:
      self.rgb_image = self.load_rgb_image()
      self.show_rgb()

  # --------------------------------------------------
  # FILE UPLOAD METHOD
  # --------------------------------------------------
  def upload_image(self):
    """Opens file dialog to let user select a GeoTIFF image."""
    file_path, _ = QFileDialog.getOpenFileName(
        self,
        "Select GeoTIFF File",
        "",
        "TIFF Files (*.tif *.TIF *.tiff);;All Files (*)",
    )

    if file_path:
      self.tif_path = file_path
      self.path_label.setText(f"Selected File: {file_path}")
      self.count_label.setText("Trees: 0")
      self.rgb_image = self.load_rgb_image()

      if not self.rgb_image.isNull():
        self.show_rgb()
        QMessageBox.information(
            self, "Success", "Image loaded successfully!"
        )
      else:
        QMessageBox.warning(
            self, "Error", "Failed to process the uploaded GeoTIFF file."
        )

  # --------------------------------------------------
  # HELPER: Percentile Stretch
  # --------------------------------------------------
  def percentile_stretch(self, arr, p_low=2, p_high=98):
    if arr.size == 0:
      return arr
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    arr = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    return (arr * 255).astype(np.uint8)

  # --------------------------------------------------
  # RGB IMAGE PROCESSING
  # --------------------------------------------------
  def load_rgb_image(self) -> QImage:
    if not self.tif_path:
      return QImage()
    try:
      with rasterio.open(self.tif_path) as src:
        b1 = src.read(1).astype(np.float32)  # Blue
        b2 = src.read(2).astype(np.float32)  # Green
        b3 = src.read(3).astype(np.float32)  # Red

      R = self.percentile_stretch(b3)
      G = self.percentile_stretch(b2)
      B = self.percentile_stretch(b1)

      rgb = np.dstack([R, G, B]).astype(np.uint8)
      h, w, _ = rgb.shape
      return QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    except Exception as e:
      print(f"Error loading image: {e}")
      return QImage()

  # --------------------------------------------------
  # DISPLAY MODES
  # --------------------------------------------------
  def show_rgb(self):
    if self.rgb_image.isNull():
      QMessageBox.warning(self, "Warning", "Please upload a valid image first.")
      return
    pixmap = QPixmap.fromImage(self.rgb_image)
    scene = QGraphicsScene()
    scene.addItem(QGraphicsPixmapItem(pixmap))
    self.viewer.setScene(scene)
    self.legend_label.setText("")

  def _compute_ndvi(self):
    with rasterio.open(self.tif_path) as src:
      red = src.read(3).astype(np.float32)
      nir = src.read(4).astype(np.float32)

    denom = nir + red
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)

  def show_ndvi_raw(self):
    if not self.tif_path:
      QMessageBox.warning(self, "Warning", "Please upload an image first.")
      return
    try:
      ndvi = self._compute_ndvi()
      ndvi_norm = ((ndvi + 1.0) / 2.0 * 255.0).astype(np.uint8)

      h, w = ndvi_norm.shape
      qimg = QImage(ndvi_norm.tobytes(), w, h, w, QImage.Format_Grayscale8)

      pixmap = QPixmap.fromImage(qimg)
      scene = QGraphicsScene()
      scene.addItem(QGraphicsPixmapItem(pixmap))
      self.viewer.setScene(scene)

      self.legend_label.setText(
          "NDVI (Grayscale):\n"
          " Dark (near -1): water / bare soil / no vegetation\n"
          " Light (near +1): dense & healthy vegetation\n"
          " Mid-gray: moderate vegetation"
      )
    except Exception as e:
      QMessageBox.critical(
          self, "Error", f"Failed to compute NDVI (ensure 4-band image): {e}"
      )

  def show_ndvi_heatmap(self):
    if not self.tif_path:
      QMessageBox.warning(self, "Warning", "Please upload an image first.")
      return
    try:
      ndvi = self._compute_ndvi()
      ndvi_norm = ((ndvi + 1.0) / 2.0 * 255.0).astype(np.uint8)

      viridis = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_VIRIDIS)
      viridis_rgb = cv2.cvtColor(viridis, cv2.COLOR_BGR2RGB)

      h, w, _ = viridis_rgb.shape
      qimg = QImage(viridis_rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)

      pixmap = QPixmap.fromImage(qimg)
      scene = QGraphicsScene()
      scene.addItem(QGraphicsPixmapItem(pixmap))
      self.viewer.setScene(scene)

      self.legend_label.setText(
          "NDVI Heatmap (Viridis):\n"
          " Dark Blue / Purple: very low NDVI (soil / water / stressed)\n"
          " Green: medium NDVI (moderate vegetation)\n"
          " Yellow: high NDVI (dense / healthy vegetation)"
      )
    except Exception as e:
      QMessageBox.critical(
          self, "Error", f"Failed to compute NDVI heatmap: {e}"
      )

  # --------------------------------------------------
  # ONNX DETECTOR LOGIC
  # --------------------------------------------------
  def prepare_image_for_onnx(self, rgb: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(
        rgb, (self.model_input_width, self.model_input_height)
    )
    img = img_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

  def run_onnx_tree_detection(self, rgb: np.ndarray):
    orig_h, orig_w, _ = rgb.shape
    input_tensor = self.prepare_image_for_onnx(rgb)

    outputs = self.ort_session.run(None, {self.ort_input_name: input_tensor})
    pred = outputs[0][0]

    cx, cy, w, h, score = pred
    score_threshold = 0.45
    iou_threshold = 0.5

    scale_x = orig_w / self.model_input_width
    scale_y = orig_h / self.model_input_height

    boxes, scores_keep = [], []

    for i in range(cx.shape[0]):
      if score[i] < score_threshold:
        continue
      x1 = (cx[i] - w[i] / 2.0) * scale_x
      y1 = (cy[i] - h[i] / 2.0) * scale_y
      x2 = (cx[i] + w[i] / 2.0) * scale_x
      y2 = (cy[i] + h[i] / 2.0) * scale_y

      if x2 <= x1 or y2 <= y1:
        continue
      boxes.append([x1, y1, x2, y2])
      scores_keep.append(float(score[i]))

    if not boxes:
      return []

    boxes = np.array(boxes, dtype=np.float32)
    scores_keep = np.array(scores_keep, dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores_keep.argsort()[::-1]

    keep = []
    while order.size > 0:
      i = order[0]
      keep.append(i)
      xx1 = np.maximum(x1[i], x1[order[1:]])
      yy1 = np.maximum(y1[i], y1[order[1:]])
      xx2 = np.minimum(x2[i], x2[order[1:]])
      yy2 = np.minimum(y2[i], y2[order[1:]])

      w_i = np.maximum(0.0, xx2 - xx1)
      h_i = np.maximum(0.0, yy2 - yy1)
      inter = w_i * h_i

      iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
      inds = np.where(iou <= iou_threshold)[0]
      order = order[inds + 1]

    return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])) for i in keep]

  def count_trees(self):
    if not self.tif_path:
      QMessageBox.warning(self, "Warning", "Please upload an image first.")
      return

    self.legend_label.setText("")
    self.count_label.setText("Processing... Please wait.")
    QApplication.processEvents()

    try:
      with rasterio.open(self.tif_path) as src:
        b1 = src.read(1).astype(np.float32)
        b2 = src.read(2).astype(np.float32)
        b3 = src.read(3).astype(np.float32)

      R = self.percentile_stretch(b3)
      G = self.percentile_stretch(b2)
      B = self.percentile_stretch(b1)

      rgb = np.dstack([R, G, B]).astype(np.uint8)
      h, w, _ = rgb.shape
      tile_size = self.model_input_width

      output = rgb.copy()
      tree_count = 0

      for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
          tile = rgb[y : y + tile_size, x : x + tile_size]
          if tile.size == 0:
            continue
          th, tw, _ = tile.shape

          if th < tile_size or tw < tile_size:
            padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded_tile[:th, :tw, :] = tile
            boxes_tile = self.run_onnx_tree_detection(padded_tile)
          else:
            boxes_tile = self.run_onnx_tree_detection(tile)

          for x1, y1, x2, y2 in boxes_tile:
            if x1 >= tw or y1 >= th:
              continue
            gx1 = max(0, min(w - 1, x + x1))
            gy1 = max(0, min(h - 1, y + y1))
            gx2 = max(0, min(w - 1, x + x2))
            gy2 = max(0, min(h - 1, y + y2))

            cv2.rectangle(output, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            tree_count += 1

      self.count_label.setText(f"Trees: {tree_count}")

      cv2.putText(
          output,
          f"Trees: {tree_count}",
          (10, 30),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.0,
          (255, 0, 0),
          2,
          cv2.LINE_AA,
      )

      h_out, w_out, _ = output.shape
      qimg = QImage(
          output.tobytes(), w_out, h_out, w_out * 3, QImage.Format_RGB888
      )
      pixmap = QPixmap.fromImage(qimg)
      scene = QGraphicsScene()
      scene.addItem(QGraphicsPixmapItem(pixmap))
      self.viewer.setScene(scene)

      QMessageBox.information(self, "Success", f"Found {tree_count} trees.")

    except Exception as e:
      self.count_label.setText("Error!")
      QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
  onnx_model_path = "tree_model.onnx"

  app = QApplication(sys.argv)
  window = ImageViewer(onnx_model_path=onnx_model_path)
  window.show()
  sys.exit(app.exec_())
