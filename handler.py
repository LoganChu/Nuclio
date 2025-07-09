import torch
import torchvision.transforms as T
import base64
import numpy as np
import cv2
from io import BytesIO

model = None
transform = T.Compose([
    T.ToTensor(),
])

def init_context(context):
    global model
    model = torch.load('/opt/nuclio/model.pth', map_location=torch.device('cpu'))
    model.eval()

def handler(context, event):
    global model
    data = event.body

    # Decode base64 image
    encoded_image = data.get("image", "")
    img_bytes = base64.b64decode(encoded_image)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Apply transform
    input_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().cpu().numpy().astype(np.uint8) * 255

    # Find contours to return as polygon
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        if contour.shape[0] >= 3:
            points = contour.squeeze().tolist()
            result.append({
                "type": "polygon",
                "points": [p for point in points for p in point],  # flatten [x, y]
            })

    return context.Response(body={
        "version": "1.0",
        "data": result
    }, headers={"Content-Type": "application/json"})
