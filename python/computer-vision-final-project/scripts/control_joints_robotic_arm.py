import sim
import sys
import cv2
import numpy as np
import time
def closeGripper():
    sim.simxSetJointTargetVelocity(clientID,motorGripper[1],0.04,sim.simx_opmode_streaming)
    sim.simxSetJointTargetPosition(clientID,motorGripper[0],getCurrentAngle(motorGripper[1])*-0.5,sim.simx_opmode_streaming)
    time.sleep(5)
def openGripper():
    sim.simxSetJointTargetVelocity(clientID,motorGripper[1],-0.04,sim.simx_opmode_streaming)
    sim.simxSetJointTargetPosition(clientID,motorGripper[0],getCurrentAngle(motorGripper[1])*-0.5,sim.simx_opmode_streaming)
    time.sleep(5)
#This function gets the angle of the joint
def getCurrentAngle(joint):
    _,angle=sim.simxGetJointPosition(clientID,joint,sim.simx_opmode_oneshot_wait)
    angle=(angle*180)/np.pi
    return angle
#This function gets as arguments joint: The joint to be moved
#theta_final: Angle (or Position) where the joint ends the movement in degrees
#step: For smoothness, a lower step makes more moves between positions. This step is also in degrees
def moveJoint(joint,theta_final,step):
    theta_ini=getCurrentAngle(joint)
    if(theta_ini>theta_final):
        i=theta_ini
        while i>=theta_final:
            print(i)
            theta_rad=(i*np.pi)/180
            sim.simxSetJointTargetPosition(clientID,joint,theta_rad,sim.simx_opmode_oneshot)
            i=i-step
    else:
        i=theta_ini
        while i<=theta_final:
            print(i)
            theta_rad=(i*np.pi)/180
            sim.simxSetJointTargetPosition(clientID,joint,theta_rad,sim.simx_opmode_oneshot)
            i=i+step
        
        
    
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

motorArm=[0,1,2,3,4]
motorError=[False,False,False,False,False]
motorGripper=[0,1]
gripperError=[False,False]
motorError[0],motorArm[0]=sim.simxGetObjectHandle(clientID,'youBotArmJoint0#2',sim.simx_opmode_blocking)
motorError[1],motorArm[1]=sim.simxGetObjectHandle(clientID,'youBotArmJoint1#2',sim.simx_opmode_blocking)
motorError[2],motorArm[2]=sim.simxGetObjectHandle(clientID,'youBotArmJoint2#2',sim.simx_opmode_blocking)
motorError[3],motorArm[3]=sim.simxGetObjectHandle(clientID,'youBotArmJoint3#2',sim.simx_opmode_blocking)
motorError[4],motorArm[4]=sim.simxGetObjectHandle(clientID,'youBotArmJoint4#2',sim.simx_opmode_blocking)

gripperError[0],motorGripper[0]=sim.simxGetObjectHandle(clientID,'youBotGripperJoint1#2',sim.simx_opmode_blocking)
gripperError[1],motorGripper[1]=sim.simxGetObjectHandle(clientID,'youBotGripperJoint2#2',sim.simx_opmode_blocking)

def move_yellow_block():
    moveJoint(motorArm[2],60,0.01)
    moveJoint(motorArm[3],62,0.01)
    moveJoint(motorArm[1],36,0.01)
    closeGripper()
    moveJoint(motorArm[0],180,0.01)
    openGripper()
    moveJoint(motorArm[0],-1.8,0.01)

def move_blue_block():
    
    moveJoint(motorArm[2],52,0.01)
    moveJoint(motorArm[3],50,0.01)
    moveJoint(motorArm[1],45,0.01)
    closeGripper()
    moveJoint(motorArm[0],180,0.01)
    openGripper()
    moveJoint(motorArm[0],-1.8,0.01)

def move_red_block():
    moveJoint(motorArm[2],35,0.01)
    moveJoint(motorArm[3],45,0.01)
    moveJoint(motorArm[1],58,0.01)
    closeGripper()
    moveJoint(motorArm[0],180,0.01)
    openGripper()
    moveJoint(motorArm[0],-1.8,0.01)

move_yellow_block()
move_blue_block()
move_red_block()
    
cv2.destroyAllWindows()
#Finalizo conexión
sim.simxFinish(-1)
