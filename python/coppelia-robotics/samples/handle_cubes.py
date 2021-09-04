# Built-in imports
import sys 

# Own imports 
import sim


def main():
    # End connexion 
    sim.simxFinish(-1)

    # Create new connexion
    clientID = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

    if (clientID != -1):
        print("Connexion OK")
    else:
        print("Fatal error - No connexion")

    cubes = [None, None, None, None, None]
    floor = None

    cube_names = ["Cuboid9", "Cuboid4", "Cuboid6", "Cuboid7", "Cuboid5"]
    for i in iter(range(5)):
        _, cubes[i] = sim.simxGetObjectHandle(
            clientID,
            cube_names[i],
            sim.simx_opmode_oneshot_wait
        )
    _, floor = sim.simxGetObjectHandle(
            clientID,
            "Floor",
            sim.simx_opmode_oneshot_wait
        )

    position = [-0.2250,0.3000,0.0925]
    sim.simxSetObjectPosition(
        clientID,
        cubes[0],
        floor,
        position,
        sim.simx_opmode_oneshot_wait
    )
    
    print("Done")

    # End connexion 
    sim.simxFinish(-1)  

if __name__ == "__main__":
    sys.exit(main())