from dotenv import load_dotenv

def register_server(server_address: str = "192.168.42.5:7077", server_token: str = None):
    import os
    if server_token is None:
        server_token = "eyJhbGciOiJFUzI1NiIsICJraWQiOiIzV3NlencifQ.eyJhdWQiOiJodHRwOi8vbG9jYWxob3N0OjgwODAiLCAiZXhwIjoxNzE4NjI2MDE5LCAiZ3JwY19icm9hZGNhc3RfYWRkcmVzcyI6ImxvY2FsaG9zdDo3MDc3IiwgImlhdCI6MTcxMDg1MDAxOSwgImlzcyI6Imh0dHA6Ly9sb2NhbGhvc3Q6ODA4MCIsICJzZXJ2ZXJfdXJsIjoiaHR0cDovL2xvY2FsaG9zdDo4MDgwIiwgInN1YiI6IjcwN2QwODU1LTgwYWItNGUxZi1hMTU2LWYxYzQ1NDZjYmY1MiIsICJ0b2tlbl9pZCI6ImJhMjA4NjA3LTk2N2UtNDhiNC1iMjUyLTJkYmM3ZjhmMGNjMiJ9.83DD-usEIfLsJFQ82BHPSwq3Gd0MqZd9BbGfMPdgARjeekaNq5M10uLqBCcHVDbH6_LSulg2GclXrhuJksu6gw"

    os.environ["HATCHET_CLIENT_HOST_PORT"] = "192.168.42.5:7077"
    os.environ["HATCHET_CLIENT_TLS_STRATEGY"] = "none"
    os.environ["HATCHET_CLIENT_TOKEN"] = server_token
    load_dotenv()

def get_hatchet(server_address: str = "192.168.42.5:7077", server_token: str = None, debug:bool = True)->"Hatchet":
    from hatchet_sdk import Hatchet

    register_server(server_address, server_token)
    return Hatchet(debug=debug)