# pyclient.py – connects Python bot to TORCS server (train / run)

import sys, socket, argparse, os, time
import driver

parser = argparse.ArgumentParser("TORCS Python DQN client")
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', type=int, default=3001)
parser.add_argument('--id',   default='SCR')
parser.add_argument('--train', action='store_true',
                    help="enable training mode (DQN)")
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--maxSteps', type=int, default=0)
args = parser.parse_args()

# ― socket setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(1.0)

# ― driver
d = driver.Driver(stage=3, train=args.train)

shutdown = False
curEp    = 0

# ― handshake
while True:
    init_msg = args.id + d.init()
    sock.sendto(init_msg.encode(), (args.host, args.port))
    try:
        buf, _ = sock.recvfrom(1000)
        if b'***identified***' in buf:
            print("Connected with ID:", args.id)
            break
    except socket.error: pass

# ― main episode loop
while not shutdown and curEp < args.episodes:
    step = 0
    episode_reward = 0.0

    while True:  # step loop
        try:
            data, _ = sock.recvfrom(1000)
            data = data.decode()
        except socket.error:
            continue

        if '***shutdown***' in data:
            shutdown = True
            print("Server shutdown.")
            break

        if '***restart***' in data:
            print("Restart message.")
            break

        # choose mode
        out_msg = d.drive(data)

        # (optional) track reward when training
        if args.train:
            episode_reward += d.reward()

        sock.sendto(out_msg.encode(), (args.host, args.port))
        step += 1
        if args.maxSteps and step >= args.maxSteps:
            break

    print(f"Episode {curEp+1}/{args.episodes}  reward={episode_reward:.2f}")
    curEp += 1

d.onShutDown()
sock.close()
