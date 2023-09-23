from PIL import Image
import os

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

# Load cấu hình từ tệp cấu hình
config = Cfg.load_config_from_file('/workspace/vietocr/config.yml')

# Tạo một đối tượng Trainer với cấu hình đã chỉnh sửa
trainer = Trainer(config, pretrained=False)

# Bắt đầu quá trình train từ đầu
trainer.train()