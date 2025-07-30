from openai import AzureOpenAI

import gobbler.constants as c
import gobbler.cred as cred
from gobbler.utils import dump_usage_data, get_usage_file, load_usage_data


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
        self.usage_file = get_usage_file(c.USAGE_AOAI_EMBEDDING)
        self.usage_data = load_usage_data(self.usage_file)
        self.usage_data[cred.AZURE_EMB_MODEL] = self.usage_data.get(
            cred.AZURE_EMB_MODEL, {c.FLD_USAGE_PROMPT: 0}
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=cred.AZURE_EMB_MODEL, input=texts
        )
        self.usage_data[cred.AZURE_EMB_MODEL][
            c.FLD_USAGE_PROMPT
        ] += response.usage.prompt_tokens
        dump_usage_data(self.usage_data, self.usage_file)
        return [x.embedding for x in response.data]
