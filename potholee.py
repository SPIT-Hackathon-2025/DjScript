from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="F3EMGUQMEz5Oj4QmIU6D"
)

result = CLIENT.infer(your_image.jpg, model_id="traffic-signs-recognition-nl4tf/1")