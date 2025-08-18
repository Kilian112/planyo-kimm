import requests, os, json
from typing import List

class OllamaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv('OLLAMA_URL')
        if not self.base_url:
            raise EnvironmentError("Required environment variable 'OLLAMA_URL' is missing.")

        if not self.llm_service_available():
            raise ConnectionError('Could not connect to Ollama. Check if service is running.')

    def send_prompt(self, prompt: str, model_name: str, system_prompt: str = None) -> str:
        if not self.is_model_available(model_name):
            raise AttributeError(f"Model '{model_name}' cannot be found in Ollama. Avaiable Models are: {self.list_avalable_models()}")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()

            result = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():
                    chunk = json.loads(line)
                    result += chunk.get("response", "")
                    if chunk.get("done", False):
                        break

            return result
        except requests.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return ""

    def embed(self, text, model_name: str) -> List[float]:
        if not self.is_model_available(model_name):
            raise AttributeError(f"Model '{model_name}' cannot be found in Ollama. Avaiable Models are: {self.list_avalable_models()}")

        payload = {
            "model": model_name,
            "prompt": text
        }
        try:
            response = requests.post(f"{self.base_url}/api/embeddings",
                                    json=payload)
            response.raise_for_status()

            data = response.json()
            embedding = data["embedding"]
            return embedding
        except requests.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return ""

    def list_avalable_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except requests.RequestException:
            return []

    def is_model_available(self, model_name: str) -> bool:
        return model_name in self.list_avalable_models()

    def llm_service_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException as e:
            print(f'Ollama connection failed because of: {e}')
            return False