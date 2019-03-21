# Script to get the system's info

from __future__ import print_function
import platform
import multiprocessing as mp

'''
    Functions:
    ------------------------------------------------------
    sysinfo(display = True, return_info = False)
        Gets the details about the system
    ------------------------------------------------------
'''

# Function to get the system's information
def sysinfo(display = True, return_info = False):
    # ----------------------------------------------------
    # INPUT:
    # ----------------------------------------------------
    # display      : bool : condition to print details
    # return_info  : bool : condition to return details
    # ----------------------------------------------------
    # OUTPUT: 
    # ----------------------------------------------------
    # info  : dict : stores the information of the system
    # ----------------------------------------------------

    # Dictionary storing the information
    info = {
        'python_version' : platform.python_version(),
        'compiler'       : platform.python_compiler(),
        'os'             : platform.system(),
        'verison'        : platform.release(),
        'machine'        : platform.machine(),
        'processor'      : platform.processor(),
        'cores'          : mp.cpu_count(),
        'interpreter'    : platform.architecture()[0]
        }

    # Displays the system details
    if display:
        print('> Python version   :', info['python_version'])
        print('> Compiler         :', info['compiler'])
        print('> Operating System :', info['os'])
        print('> Version          :', info['verison'])
        print('> Machine          :', info['machine'])
        print('> Processor        :', info['processor'])
        print('> CPU count        :', info['cores'])
        print('> Interpreter      :', info['interpreter'])

    # Returns the system info
    if return_info:
        return info
    
if __name__ == '__main__':
    sysinfo()
