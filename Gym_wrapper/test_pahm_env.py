import gymnasium as gym
from learned_pahm import LearnedPAHM
import numpy as np

env = LearnedPAHM(render_mode="human")

state="Sine"


while True:
    observation,info = env.reset(seed=123)
    done=False
    step=0
    
    while not done:
        if state=="Random":
            action = env.action_space.sample()*0.25
        elif state=="Off":
            action = np.array([0.0])
        elif state=="Step":
            action = np.array([0.2])
        elif state=="Sine":
            action = np.array([0.25*(1+np.sin(step*2*np.pi/(50*3)))/2])

        if np.random.rand() < 1.0/(50*3): # roughtly every 3 seconds change...
            state=["Random","Off","Step","Sine"][np.random.randint(0,4)]
            print(f"State={state}")
        
            
        observation,reward,terminated,truncated,info = env.step(action)
        if step%5==0:
            env.render() 
        step+=1
        done=terminated or truncated

        if step%50 == 0:
            print(".")

env.close()
