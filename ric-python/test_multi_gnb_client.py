import socket
import json
import numpy as np


def build_test_sinr(num_gnbs=3, num_beams=4, num_ues=5, seed=42):
    np.random.seed(seed)
    # Random but reasonable SINR values in dB
    sinr = np.random.uniform(-10, 30, size=(num_gnbs, num_beams, num_ues))
    return sinr.tolist()


def main():
    host = "127.0.0.1"
    port = 5556  # test server port

    num_gnbs = 3
    num_beams = 4
    num_ues = 5

    sinr_matrix = build_test_sinr(num_gnbs, num_beams, num_ues)

    request = {
        "scenario_id": 1,
        "num_gnbs": num_gnbs,
        "num_ues": num_ues,
        "sinr_matrix_dB": sinr_matrix,
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    req_json = json.dumps(request)
    sock.sendall(req_json.encode("utf-8"))

    resp_data = sock.recv(65536)
    sock.close()

    response = json.loads(resp_data.decode("utf-8"))

    print("Algorithm:", response.get("algorithm"))
    print("Objective:", response.get("objective_value"))
    print("gNB for UE:", response.get("gnb_for_ue"))
    print("Beam for UE:", response.get("beam_for_ue"))


if __name__ == "__main__":
    main()
