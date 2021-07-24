import json
import subprocess
import threading
import platform
import time
import shlex
from subprocess import Popen, PIPE, STDOUT
import websocket

from gym_game.envs.models.physics.physics import ensure_bounds



# This class connects to the ws to controll the Attacker robot. ITS IMPEATIVE that you call '<instance>.ws.close()' after initiating an instance of this class
class GameControls:

    def __init__(self):
        self.defender_position = [30, 30]
        self.attacker_position = [30, 30]
        self.delays = {1, 5000, 10000, 15000, 20000, 25000, 30000}
        self._trajectory = [30, 30]
        self.x_min = 10
        self.x_max = 80
        self.ping = 0
        self.last_message_time = time.time()
        self.last_hit_time = round(time.time())
        self.hits_since_start = 0
        self.score = 20
        self.done = False
        self.level = 1
        self.y_min = 10
        self.y_max = 80
        self.uri = "wss://robot.comnets.net:5001/"
        self.step = 2
        self.role = "attacker"
        self.is_open = False
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(self.uri,
                                         on_message=lambda ws, msg: self.on_message(ws, msg),
                                         on_error=lambda ws, msg: self.on_error(ws, msg),
                                         on_close=lambda ws: self.on_close(ws),
                                         on_open=lambda ws: self.on_open(ws))
        self.listenerThread = threading.Thread(target=self.ws.run_forever)
        self.await_for_connection()
        self.update_ping()


    def __del__(self):
        self.ws.close()
        self.listenerThread.join(1)

    def await_for_connection(self):
        timeout = time.time() + 3  # 3 seconds from now
        self.listenerThread.start()
        print("Connecting to WS...")
        while not self.is_open:
            if time.time() > timeout:
                raise Exception("Couldnt connect to socket")
            time.sleep(0.1)

    def on_message(self, ws, message):
        try:
            self.last_message_time = time.time()
            msg = json.loads(message)
            self.parse_msg(msg)
        except Exception as e:
            print(e)
            print(message)

    def on_error(self, ws, error):
        print("ERROR: {}".format(error))
        raise Exception("{}".format(error))

    def on_close(self, ws):
        self.moveToStartPosition()
        self.is_open = False

    def on_open(self, ws):
        self.is_open = True
        self.moveToStartPosition()

    @property
    def robot_position(self):
        return self._robot_position

    @robot_position.setter
    def robot_position(self, robot_pos):
        if robot_pos:
            self._robot_position = [ensure_bounds(robot_pos[0], self.x_min, self.x_max),
                                    ensure_bounds(robot_pos[1], self.y_min, self.y_max)]
            self.sendCommand("setPosition", self.robot_position)

    # def onmessage
    def sendCommand(self, command, data="", ws=None):
        if self.last_message_time - time.time() >= 3: 
            raise Exception("Connection to Web Socket is terminated")
        message = {"command": command, "data": data, "level": self.level}
        if ws:
            ws.send(json.dumps(message))
        else:
            self.ws.send(json.dumps(message))

    def moveTo(self, position):
        self.robot_position = position

    def moveLeft(self):
        self.robot_position = [self.robot_position[0] - self.step, self.robot_position[1]]

    def moveRight(self):
        self.robot_position = [self.robot_position[0] + self.step, self.robot_position[1]]

    def moveUp(self):
        self.robot_position = [self.robot_position[0], self.robot_position[1] - self.step]

    def moveDown(self):
        self.robot_position = [self.robot_position[0], self.robot_position[1] + self.step]

    def fire(self):
        print("FIREEEEEEEEEEEEEe")
        self.sendCommand("Fire", "attacker")

    # Returns 2 arrays with the form [Attacker_x, Attacker_y], [Defender_x, Defender_y]
    def getRobotsPositions(self):
        return self.attacker_position, self.defender_position

    def moveToStartPosition(self):
        self.robot_position = [30, 30]


    def parse_msg(self, msg):
        if msg['command'] == 'robot_position':
            self.defender_position = [msg['data'][0], msg['data'][1]]
            self.attacker_position = [msg['data'][2], msg['data'][3]]
            #print('robot_positions: Attacker: {}, Defender: {}'.format(self.attacker_position, self.defender_position))
       #this is suposed to lower the error rate on the msg hitTarget command that is being sent by the server 
        elif msg['command'] == 'hitTarget' :
            print(msg)
            hit_radius = 12
            xpos = abs(self.defender_position[0]-self.attacker_position[0]) 
            ypos = abs(self.defender_position[0]-self.attacker_position[0])
            if time.time() - self.last_hit_time > 3 and xpos <= hit_radius and  ypos <= hit_radius :
                print(msg)
                print("TARGET HIT!!! REJOICE!!!")
                self.hits_since_start += 1
                self.last_hit_time = round(time.time())
            elif xpos > 10:
                print("Hit command rejected")
                print("xpos: {}".format(xpos))
            elif ypos > 10:
                print("Hit command rejected")
                print("ypos: {}".format(ypos))


    def get_simple_cmd_output(self, cmd, stderr=STDOUT):
        args = shlex.split(cmd)
        return Popen(args, stdout=PIPE, stderr=stderr).communicate()[0].decode()

    def update_ping(self):
        threading.Timer(1 , self.update_ping).start()
        #host of server: robo
        host = "141.30.32.22"
        if platform.system().lower() != 'windows':
            cmd = "fping {host} -C 3 -q".format(host=host)
            res = [float(x) for x in self.get_simple_cmd_output(cmd).strip().split(':')[-1].split() if x != '-']
            if len(res) > 0:
                self.ping = sum(res) / len(res)
        else:
            cmd = "fping {host} -n 3".format(host=host)
            avg = float(self.get_simple_cmd_output(cmd).strip().split(':')[-1].split()[10])/1000
            self.ping = avg

# gc = GameControls()
# print(gc.robot_position)
# do_some_movement()
# gc.ws.close()
# del gc


# ws = websocket.create_connection("wss://robot.comnets.net:5001/")
# ws.send(sendCommand("setPosition", [30,30]))


# import asyncio
# import websocket
# import json
# import asyncio

# class GameControls:
#    def __init__(self):
#        self.uri = "wss://robot.comnets.net:5001/"
#        self.level = 1
#        self.ws = websocket.WebSocketApp(self.uri,
#                    on_message = lambda ws,msg: self.on_message(ws, msg),
#                    on_error   = lambda ws,msg: self.on_error(ws, msg),
#                    on_close   = lambda ws:     self.on_close(ws),
#                    on_open    = lambda ws:     self.on_open(ws))
#
#    def on_message(ws, message):
#        print(message)
#
#    def on_error(ws, error):
#        print (error)
#
#    def on_close(ws):
#        print ("### closed ###")

#    def on_open(self,ws):
#        print("Hello King Baboon")
#        self.ws.send("Hello King Baboon")
#
#    def sendCommand(self, command, data=""):
#        message = {}
#        message["command"] = command
#        message["data"] = data
#        message["level"] = self.level
#        self.ws.send(json.dumps(message))
#
#    websocket.enableTrace(True)
#    gamecontrols = GameControls()
#    gamecontrols.ws.run_forever()
#    gamecontrols.sendCommand(command="setPosition",data=[30,30])
