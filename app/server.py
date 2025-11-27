from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from .rag_pipeline import RagChatbot, RagSettings

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_app() -> Flask:
    settings = RagSettings()
    chatbot = RagChatbot(settings)

    app = Flask(
        __name__,
        static_folder=str(Path(__file__).resolve().parent.parent / "static"),
        template_folder=str(Path(__file__).resolve().parent.parent / "templates"),
    )
    app.config["chatbot"] = chatbot

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/chat", methods=["POST"])
    def chat():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "질문을 입력해주세요."}), 400
        try:
            result = chatbot.answer(question)
            return jsonify(result)
        except Exception as exc:  # pragma: no cover - logged for debug
            logger.exception("RAG pipeline error: %s", exc)
            return jsonify({"error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

