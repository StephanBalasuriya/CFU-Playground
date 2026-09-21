#!/bin/bash

UART="/tmp/mnv2_uart"

if [ ! -e "$UART" ]; then
    echo "ERROR: UART PTY does not exist."
    echo
    echo "Start Renode first using:"
    echo "  ./run.sh"
    echo
    echo "Then, inside Renode, run:"
    echo '  emulation CreateUartPtyTerminal "term" "/tmp/mnv2_uart" true'
    echo '  connector Connect sysbus.uart term'
    exit 1
fi

echo "=== Connecting to CFU Playground UART ==="
echo "UART: $UART"
echo "Press Ctrl+] to exit."
echo

pyserial-miniterm "$UART" 115200
