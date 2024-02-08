# ************************************************************************************************************************* #
#                                           Copyright (C) 2022 Jorge Brenes Alfaro.
#                                           EL5617 Trabajo Final de Graduación.
#                                           Escuela de Ingeniería Electrónica.
#                                           Tecnológico de Costa Rica.
# *************************************************************************************************************************** #

#   This file is responsible for collecting the PAHM data. To do this, this file
#   makes use of the TransmitReceive and RepeatedTimer classes, which are located in
#   their respective files. In addition, it is possible to use two inputs: the manual input
#   and playback input to replay the same previous input.

#   Libraries.
import argparse
import numpy as np
import pandas as pd

from RepeatedTimer import *
from Transmit_and_Receive import *

parser = argparse.ArgumentParser(description='Collect data from PAHM plant.')
parser.add_argument('--input',type=str,default="",help='Name of the input file to playback. ')
parser.add_argument('--output',type=str,default="",help='Name of the output file where the data is saved.')
args = parser.parse_args()

playbackMode = (args.input != "")
storeOutput  = (args.output != "")

#   Port serial definition.
serial_port  = serial.Serial("/dev/ttyTHS2", baudrate = 115200, stopbits = serial.STOPBITS_ONE,
                             bytesize = serial.EIGHTBITS, parity = serial.PARITY_NONE)

time.sleep(0.5) #   Time for pin assignment.

PAHM = TransmitReceive(serial_port)

if playbackMode:
    #       If playback mode is active, use a .csv file.
    data = pd.read_csv(args.input,delimiter=',')
    PAHM.pwm_set_safe_value(data.values[:,2])
else:
    PAHM.pwm_set_safe_value(0.0)

print('************ Starting *************\n', flush=True)
PAHM.reset()    #   Clean transmit and receive buffers.
print('Data Colecting ...\n', flush=True)
rt = RepeatedTimer(0.02, PAHM.Transmit_Receive) # No need of rt.start()

def comando(valor):
    PAHM.pwm_set_safe_value(int(valor)/1000.0)

try:
    #   PLAYBACK
    if playbackMode:
          time.sleep(np.shape(PAHM.pwm)[0]*0.02)
    else:
    #   MANUAL
        master = tkinter.Tk()
        master.title('My PWM value')
        master.geometry('500x100') 
        w = tkinter.Scale(master, from_=0, to=1000, orient=tkinter.HORIZONTAL,length=800,command=comando,bg='white', fg='black', width=20)
        w.pack()
        tkinter.mainloop()
        
        print("Exiting Program...")
        
    if storeOutput:
       PAHM.csv_doc(args.output) #  Save data in .csv file


except KeyboardInterrupt:
    print("Exiting Program...")
    PAHM.reset()
    PAHM.turn_off()

except Exception as exception_error:
    print("Error occurred.")
    print("Error: " + str(exception_error))
    PAHM.reset()
    PAHM.turn_off()
    
finally:
    rt.stop()
    PAHM.turn_off()
    print('************ Finished *************', flush=True)
    pass
