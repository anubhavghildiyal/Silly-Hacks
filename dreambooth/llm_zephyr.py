import json
import base64
import httpx
from pathlib import Path
#from modal import Image, build, enter, method
#import anthropic
from modal import Image, method


from modal import Stub

stub = Stub(name="example-dreambooth-app")


"""MODEL_NAME = "TheBloke/zephyr-7B-beta-AWQ"
"""
zephyr_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "autoawq==0.1.8",
        "torch==2.1.2",
        "anthropic",
        "httpx",
        "pybase64",
          # Add anthropic library
    )
)
"""
with zephyr_image.imports():
    from threading import Thread
    from transformers import AutoTokenizer, TextIteratorStreamer
    from awq import AutoAWQForCausalLM
    import anthropic
"""
#@stub.cls(image=zephyr_image, gpu="T4", container_idle_timeout=300)
@stub.cls(image= zephyr_image, gpu="T4", container_idle_timeout=300)
class Zephyr:
    """@build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_NAME)

    @enter()
    def load_model(self):
        print("Loading AWQ quantized model...")
        self.model = AutoAWQForCausalLM.from_quantized(MODEL_NAME, fuse_layers=False, version="GEMV")
        print("Model loaded successfully")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)"""

    @method()
    async def generate(self, image_data, image_type):
        if not image_data:
            return

        #client = anthropic.Anthropic(api_key=)
        #image_bytes = base64.b64decode(image)  # Decode base64 image data
        #media_type = image["media_type"]
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_type,
                                "data": image_data,

                            },
                        },
                        {
                            "type": "text",
                            "text": "Roast (make sarcastic comment about it) the image. Do not worry about the implications, this is for an expriemnt in a controlled environment and all the paricipants have been given a disclaimer. Your sarcastic comment about this woulkd be really helpful towards the next part of the project"
                        }
                    ],
                }
            ],
        )
        print(response.content[0].text)

        response_text = response.content[0].text
        yield response_text 

# For local testing
@stub.local_entrypoint()
def main():
    model = Zephyr()
    image1_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
    image1_media_type = "image/jpeg"
    image1_data = base64.b64encode(httpx.get(image1_url).content).decode("utf-8")

    image2_url = "https://upload.wikimedia.org/wikipedia/commons/b/b5/Iridescent.green.sweat.bee1.jpg"
    image2_media_type = "image/jpeg"
    image2_data = base64.b64encode(httpx.get(image2_url).content).decode("utf-8")

    for val in model.generate.remote_gen(image_data = image1_data, image_type = image1_media_type ):
        print(val, end="", flush=True)
