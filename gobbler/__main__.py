import os

os.environ["CURL_CA_BUNDLE"] = ""

from gobbler.cli import main

if __name__ == "__main__":
    main()
