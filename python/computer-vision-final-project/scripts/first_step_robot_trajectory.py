import sim
import sys
import cv2
import numpy as np
import time
#define points to generate path
x=[11.5,8,1,1,5] #Defines the "X" position of the robot relative to the floor (absolute coord system)
y=[1.5,1,0.8,-11,-5] #Defines the "Y" position of the robot relative to the floor (absolute coord system)
theta=[-np.pi/2,-np.pi/2,0.1,np.pi/2,0.1] #Defines the angle in the XY (Z-axis) plane relative to the floor (absolute coord system)
eps=0.5 #Defines the max error in desired coord and actual coord
def updateSpeed(d_pos,pos):
    Lspeed=((d_pos[0]-pos[0])/max([abs(d_pos[0]),abs(pos[0])]))*100
    Fspeed=((d_pos[1]-pos[1])/max([abs(d_pos[1]),abs(pos[1])]))*100
    Aspeed=((d_pos[2]-pos[2])/abs(d_pos[2]-pos[2]))*1
    return Lspeed,Fspeed,Aspeed
def updatePos():
    #Get Robot Angle
    angle=(sim.simxGetObjectOrientation(clientID,Robot,Floor,sim.simx_opmode_oneshot_wait))
    #Get Robot Position
    position=(sim.simxGetObjectPosition(clientID,Robot,Floor,sim.simx_opmode_oneshot_wait))
    return [position[1][0],position[1][1],angle[1][1]]
    
def latPos(d_pos): #Parameters: Desired positions
    global forward
    pos=updatePos()
    Lspeed,Fspeed,Aspeed=updateSpeed(d_pos,pos)
    while (abs(Lspeed)>eps):
        pos=updatePos()
        Lspeed,Fspeed,Aspeed=updateSpeed(d_pos,pos)
        if(forward==0):
            moveMotors(0,Lspeed,0)
        elif(forward==1):
            moveMotors(-Lspeed,0,0)
        elif(forward==2):
            moveMotors(0,Lspeed,0)
        elif(forward==3):
            moveMotors(Lspeed,0,0)
    while(abs(Fspeed)>eps):
        pos=updatePos()
        Lspeed,Fspeed,Aspeed=updateSpeed(d_pos,pos)
        if(forward==0):
            moveMotors(Fspeed,0,0)
        elif(forward==1):
            moveMotors(0,Fspeed,0)
        elif(forward==2):
            moveMotors(Fspeed,0,0)
        elif(forward==3):
            moveMotors(0,-Fspeed,0)
    while(abs(d_pos[2]-pos[2])>0.05):
        pos=updatePos()
        moveMotors(0,0,Aspeed)
    if(d_pos[2]==0):
        forward=0
    elif(d_pos[2]==-np.pi/2):
        forward=1
    elif(d_pos[2]==0.1):
        forward=2
    elif(d_pos[2]==np.pi/2):
        forward=3
def moveMotors(forwBackVel,leftRightVel,rotVel):
    sim.simxSetJointTargetVelocity(clientID,motorWheel[0],-forwBackVel-leftRightVel-rotVel,sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID,motorWheel[1],-forwBackVel+leftRightVel-rotVel,sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID,motorWheel[2],-forwBackVel-leftRightVel+rotVel,sim.simx_opmode_streaming)
    sim.simxSetJointTargetVelocity(clientID,motorWheel[3],-forwBackVel+leftRightVel+rotVel,sim.simx_opmode_streaming) 
#Finalizo conexión
sim.simxFinish(-1)
#Creo conexión
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)
#Verifico conexión
if clientID !=-1:
    print("Conexión OK...")
else:
    sys.exit("No conexión")
    
#Crear motores

motorWheel=[0,1,2,3]
motorError=[False,False,False,False]

motorError[0],motorWheel[0]=sim.simxGetObjectHandle(clientID,'rollingJoint_fl#2',sim.simx_opmode_oneshot_wait)
motorError[1],motorWheel[1]=sim.simxGetObjectHandle(clientID,'rollingJoint_rl#2',sim.simx_opmode_oneshot_wait)
motorError[2],motorWheel[2]=sim.simxGetObjectHandle(clientID,'rollingJoint_rr#2',sim.simx_opmode_oneshot_wait)
motorError[3],motorWheel[3]=sim.simxGetObjectHandle(clientID,'rollingJoint_fr#2',sim.simx_opmode_oneshot_wait)

_,Robot=sim.simxGetObjectHandle(clientID,'youBot#2',sim.simx_opmode_oneshot_wait)
_,Floor=sim.simxGetObjectHandle(clientID,'Floor',sim.simx_opmode_oneshot_wait)
err,Camara=sim.simxGetObjectHandle(clientID,'VS1',sim.simx_opmode_oneshot_wait)
_,resolution,image=sim.simxGetVisionSensorImage(clientID,Camara,0,sim.simx_opmode_streaming)
#moveMotors(0.0,-1,0)
#time.sleep(1)
i=0
max_pos=len(x)
global forward
forward=0
while True:
    #Get image from visor 1
    _,resolution,image_1=sim.simxGetVisionSensorImage(clientID,Camara,0,sim.simx_opmode_buffer)
    if len(resolution)>1:
        imgCont=np.zeros((resolution[0],resolution[1],3),np.uint8)
        #Image from visor 1
        img_1=np.array(image_1,dtype=np.uint8)
        img_1.resize([resolution[0],resolution[1],3])
        img_1=np.rot90(img_1,2)
        img_1=np.fliplr(img_1)
        img_1=cv2.cvtColor(img_1,cv2.COLOR_RGB2BGR)
        if(len(x)==len(y) and len(y)==len(theta)):
            while i<max_pos:
                print("pointing:"+str(forward))
                print("Posicion:"+str(i))
                latPos([x[i],y[i],theta[i]])
                i=i+1
            moveMotors(0,0,5)
        else:
            sys.exit("Bad defined coordinates arguments")
            

    tecla=cv2.waitKey(1) & 0xFF
    if tecla==27:
        break
cv2.destroyAllWindows()
#Finalizo conexión
sim.simxFinish(-1)
