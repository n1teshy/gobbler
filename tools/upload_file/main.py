import os
import uuid
import argparse
import globals
from flask import Flask, send_file, request

app = Flask(__name__)
output_folder = None

program = argparse.ArgumentParser(
    formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100)
)
program.add_argument("-p", "--port", type=int, default=5000, dest="port", help="port")
program.add_argument("-o", "--out", type=str, default=".", dest="output_folder", help="output folder")
program.add_argument("-mf", "--make-folder", action="store_true", dest="make_folder")


@app.route("/")
def upload_file():
    return send_file("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"message": "Send file brudder, where yo file?"}, 422
    file = request.files["file"]
    filename = file.filename
    if not filename:
        content_type = request.headers("Content-Type", None)
        if content_type is None or not content_type.split("/")[1]:
            return {"message": "No filename, no content-type, what are you sending?"}, 422
        filename = f"{uuid.uuid4()}.{content_type.split('/')[1]}"
    file.save(os.path.join(globals.output_folder, filename))
    return {"message": "Done, don't forget to thank me."}


if __name__ == "__main__":
    args = program.parse_args()
    output_folder = os.path.abspath(args.output_folder)
    if not os.path.exists(output_folder):
        if not args.make_folder:
            raise FileNotFoundError(f"'{output_folder}' does not exist")
        os.makedirs(args.output_folder)
    globals.output_folder = output_folder
    app.run(host="0.0.0.0", port=args.port)
