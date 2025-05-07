import sys
import argparse
import socket
import driver
import pandas as pd
import os
import keyboard  # Library to capture keyboard input

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

arguments = parser.parse_args()


try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:  
    print('Could not make a socket:', msg)
    sys.exit(-1)

# one second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

verbose = True

d = driver.Driver(arguments.stage)

while not shutdownClient:
    while True:
        print('Sending id to server: ', arguments.id)
        buf = arguments.id + d.init()

        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))  # Encode string before sending
        except socket.error as msg:  # FIXED
            print("Failed to send data...Exiting...", msg)
            sys.exit(-1)

        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()  # Decode received bytes
        except socket.error as msg:  # FIXED
            print("Didn't get response from server...", msg)

        if buf and '***identified***' in buf:
            print('Received: ', buf)
            break

    currentStep = 0

    while True:
        # wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()  # Decode received bytes
        except socket.error as msg:  # FIXED
            print("Didn't get response from server...", msg)

        if verbose and buf:
            print('Received: ', buf)

        if buf and '***shutdown***' in buf:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break

        if buf and '***restart***' in buf:
            d.onRestart()
            print('Client Restart')
            break

        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf:
                
                # Function For Automated Driving
                buf = d.drive(buf) 

                # Function For Manual Arrow Keys Driving
                #buf = d.drive_manual(buf)

        else:
            buf = '(meta 1)'

        if verbose:
            print('Sending: ', buf)

        if buf:
            try:
                sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))  # Encode before sending
            except socket.error as msg:  # FIXED
                print("Failed to send data...Exiting...", msg)
                sys.exit(-1)

    curEpisode += 1

    if curEpisode == arguments.max_episodes:
        shutdownClient = True


sock.close()
