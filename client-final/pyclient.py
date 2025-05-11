import sys
import argparse
import socket
import driver

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
parser.add_argument('--host', default='localhost', help='Host IP address (default: localhost)')
parser.add_argument('--port', type=int, default=3001, help='Host port number (default: 3001)')
parser.add_argument('--id', default='SCR', help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', type=int, default=1, help='Maximum number of episodes (default: 1)')
parser.add_argument('--maxSteps', type=int, default=0, help='Maximum number of steps (default: 0)')
parser.add_argument('--track', default=None, help='Name of the track')
parser.add_argument('--stage', type=int, default=3, help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

args = parser.parse_args()

# Print summary
print(f'Connecting to server host ip: {args.host} @ port: {args.port}')
print(f'Bot ID: {args.id}')
print(f'Maximum episodes: {args.maxEpisodes}')
print(f'Maximum steps: {args.maxSteps}')
print(f'Track: {args.track}')
print(f'Stage: {args.stage}')
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print(f'Could not make a socket: {msg}')
    sys.exit(-1)

sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

d = driver.Driver(stage=args.stage)

while not shutdownClient:
    while True:
        init_msg = args.id + d.init()
        if args.verbose:
            print(f'Sending id to server: {args.id}')
            print(f'Sending init string to server: {init_msg}')
        
        try:
            sock.sendto(init_msg.encode(), (args.host, args.port))
        except socket.error as msg:
            print(f'Failed to send data: {msg}...Exiting...')
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error:
            print("Didn't get response from server...")
            continue
    
        if '***identified***' in buf:
            print(f'Received: {buf}')
            break

    currentStep = 0
    
    while True:
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error:
            print("Didn't get response from server...")
            continue
        
        if args.verbose:
            print(f'Received: {buf}')
        
        if '***shutdown***' in buf:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        
        if '***restart***' in buf:
            d.onRestart()
            print('Client Restart')
            break
        
        currentStep += 1
        if args.maxSteps and currentStep >= args.maxSteps:
            buf = '(meta 1)'
        else:
            buf = d.drive(buf)
        
        if args.verbose:
            print(f'Sending: {buf}')
        
        try:
            sock.sendto(buf.encode(), (args.host, args.port))
        except socket.error as msg:
            print(f'Failed to send data: {msg}...Exiting...')
            sys.exit(-1)
    
    curEpisode += 1
    print(f'Episode {curEpisode}/{args.maxEpisodes} completed')
    
    if curEpisode >= args.maxEpisodes:
        shutdownClient = True

sock.close()