# Parallel algorithms of data analysis and synthesis
## Laboratory work 2

See code in file [task2.cpp](task2.cpp).

OpenMPI for Ubuntu 18.04 was installed and configured using [this instruction](https://medium.com/@li.nguyen_15905/setting-up-vscode-for-mpi-programming-b6665da6b4ad).

Compile and run the code:
```
mpicc -o task2.bin task2.cpp -lm
mpirun -np 6 ./task2.bin
```

(c) Anastasia Miroshnikova
