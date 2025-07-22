from openai import AzureOpenAI

import core.cred as cred


class AzureEmbedder:
    def __init__(self):
        if cred.AZURE_EMB_KEY is None:
            raise EnvironmentError("Missing Azure embedding key")
        self.client = AzureOpenAI(
            api_key=cred.AZURE_EMB_KEY,
            azure_endpoint=cred.AZURE_EMB_BASE,
            azure_deployment=cred.AZURE_EMB_DEPLOYMENT,
            api_version=cred.AZURE_EMB_VERSION,
        )

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=cred.AZURE_EMB_MODEL, input=[text], dimensions=3072
        )
        return response.data[0].embedding
