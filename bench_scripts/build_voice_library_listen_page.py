from common_voice_library import DEFAULT_OUTPUT_ROOT, build_listen_page


if __name__ == "__main__":
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_listen_page(DEFAULT_OUTPUT_ROOT)
    print(DEFAULT_OUTPUT_ROOT / "listen.html")
