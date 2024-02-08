/* ========================================
 *
 * Copyright Jorge Brenes Alfaro, 2022
 * EL5617 Trabajo Final de Graduación.
 * Escuela de Ingeniería Electrónica.
 * Tecnológico de Costa Rica.

 * UART COMUNICATION FROM THE PSOC TO THE JETSON TX2 AND VICE VERSA
 *
 * This file is responsible for communication between the PSOC and 
 * the Jetson TX 2. Where the Jetson TX2 sends a 2 bytes data 
 * corresponding to the PWM value. The PSoC transmits this value 
 * to the PAHM and obtains an angle via the quadrature decoder, 
 * which angle will be transmitted to the jetson tx2 in 4 bytes.
 * 
 * ========================================
*/
#include <project.h>
#include "stdio.h"
#include <stdbool.h>

uint16 pwm_value; // Receive value to write on PWM.
int angle_Quad; // Reading the quadrature decoder.
bool runcode = false; // To start/stop the process into the interrupt. It can be turn on/off in the principal switch(case).

CY_ISR_PROTO(isr_Control_Handler);
CY_ISR(isr_Control_Handler){
    if (runcode == true){
        CyPins_SetPin(LED_EJECUCION_0); // Turns on at interrupt start (Active low)
        PWM_1_WriteCompare(pwm_value); // Write the PWM value.
        angle_Quad = QuadDec_1_GetCounter(); // Read the decoder's value.
        CyPins_ClearPin(LED_EJECUCION_0); // Turns off at interrupt stop.
    }
}
                    
int main(void)
{
    CyGlobalIntEnable; // Enable global interrupts.
    
    /**** START COMPONENTS ****/
    UART_1_Start();
    PWM_1_Start();
    QuadDec_1_Start();
    Timer_1_Start();
    isr_control_StartEx(isr_Control_Handler);
    
    /**** CLEAR TRANSMIT/RECEIVE BUFFER ****/
    UART_1_ClearRxBuffer(); 
    UART_1_ClearTxBuffer(); 
    
    //**** AUXILIARY VARIABLES ****/
    int32 val;
    int8 aux; 
    uint8* arr = (uint8*)&val;
    uint16 lsb,msb;
    
    
    runcode = true; // Start the interrupts.
    
    for (;;){
        // Receive a 2-byte data for the PWM
        msb = UART_1_ReadRxData(); // Read most significant bit from the serial port.
        lsb = UART_1_ReadRxData(); // Read lowest significant bit from the serial port.
        pwm_value = lsb + (msb << 8); // PWM value (2 bytes).
        
        // Send the angular value
        val = angle_Quad; // Asign the quad_dec value.
        
        /**** REORDER THE BYTES TO TRASNMIT ****/
        aux = arr[3];
        arr[3] = arr[0];
        arr[0] = aux;
        
        aux = arr[1];
        arr[1] = arr[2];
        arr[2] = aux;
        
        UART_1_PutArray((uint8*)arr,4); // Transmitting the quad_dec value.
        CyDelay(20);
    }
}