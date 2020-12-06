#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

// HelloWorld
/*
int main(int argc, char* argv[]) {
    int ProcNum, ProcRank, RecvRank;
    MPI_Status Status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    
    if (ProcRank == 0) {
        printf("Hello from process %3d\n", ProcRank);

        for (int i = 1; i < ProcNum; i++) {
            MPI_Recv(&RecvRank, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
            printf("Hello from process %3d\n", RecvRank);
        }
    }
    else {
        MPI_Send(&ProcRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
*/

// Example with Ssend
/*
int main(int argc, char *argv[]) {
    int rank, size, i, tag = 0;
    int ProcNum, ProcRank, RecvRank;
    int buffer[10];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcRank == 0) {
        for (i=0; i<10; i++)
            buffer[i] = i;
        MPI_Ssend(buffer, 10, MPI_INT, 1, tag, MPI_COMM_WORLD);
    }
    if (ProcRank == 1) {
        for (i=0; i<10; i++)
            buffer[i] = -1;
        MPI_Recv(buffer, 10, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        for (i=0; i<10; i++) {
            if (buffer[i] != i)
                printf("Error: buffer[%d] = %d but is expected to be %d\n", i, buffer[i], i);
            else 
                printf("Got message %d\n", i);
        }
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
*/

// Example with MPI_Bsend
/*
int main(int argc, char *argv[]) {
    int *buffer;
    int ProcNum, ProcRank, buffsize = 1, TAG = 0;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcRank == 0) {
        buffer = (int *) malloc(buffsize + MPI_BSEND_OVERHEAD);
        MPI_Buffer_attach(buffer, buffsize + MPI_BSEND_OVERHEAD);
        buffer = (int *) 10;
        MPI_Bsend(&buffer, buffsize, MPI_INT, 1, TAG, MPI_COMM_WORLD);
        MPI_Buffer_detach(&buffer, &buffsize);
    }
    if (ProcRank == 1) {
        MPI_Recv(&buffer, buffsize, MPI_INT, 0, TAG, MPI_COMM_WORLD, &status);
        printf("received: %i\n", buffer);
    }
    MPI_Finalize();
    return 0;
}
*/

// Example with MPI_Sendrecv
/*
int main(int argc, char** argv){
    int numtasks = 2, ProcNum, ProcRank, next, prev, tag1 = 1, tag2 = 2;
    MPI_Request sbuf[2], rbuf[2];
    MPI_Status status_1, status_2;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    prev = ProcRank - 1;
    next = ProcRank + 1;
    if (ProcRank == 0)
        prev = numtasks - 1;
    if (ProcRank == (numtasks - 1))
        next = 0;
    MPI_Sendrecv(&sbuf[0], 1, MPI_FLOAT, prev, tag2, &rbuf[0], 1, MPI_FLOAT, next, tag2, MPI_COMM_WORLD, &status_1);
    MPI_Sendrecv(&sbuf[1], 1, MPI_FLOAT, next, tag1, &rbuf[1], 1, MPI_FLOAT, prev, tag1, MPI_COMM_WORLD, &status_2);

    printf("Node %d: all ok!\n", ProcRank);
    MPI_Finalize();
    return 0;
}
*/

// Example with continuous construction
/*
int main(int argc, char** argv) {
    int ProcRank, ProcNum, mTag = 0;
    struct { int x;
        int y;
        int z;
        } point;
    MPI_Datatype ptype;
    MPI_Status status;
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Type_contiguous(3, MPI_INT, &ptype);
    MPI_Type_commit(&ptype);
    if (ProcRank == 1) {
        point.x = 45; point.y = 36; point.z = 0;
        MPI_Send(&point, 1, ptype, 0, mTag, MPI_COMM_WORLD);
    }
    if (ProcRank == 0) {
        MPI_Recv(&point, 1, ptype, 1, mTag, MPI_COMM_WORLD, &status);
        printf("Proc No%d received point with coords (%d; %d; %d)\n", ProcRank, point.x, point.y, point.z);
    }
    MPI_Finalize(); 
}
*/

// Example with vector construction

int main(int argc, char** argv) {
    int ProcRank, ProcNum, i, j, mTag = 0;
    double x[4][8];
    MPI_Datatype coltype;
    MPI_Status status;
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Type_vector(4, 1, 8, MPI_DOUBLE, &coltype);
    MPI_Type_commit(&coltype);
    if (ProcRank == 1) {
        for (i = 0; i < 4; ++i) {
            for (j = 0; j < 8; ++j) {
                x[i][j] = pow(10.0, i + 1) + j;
            }
        }
        MPI_Send(&x[0][3], 1, coltype, 0, mTag, MPI_COMM_WORLD);
    }
    if (ProcRank == 0) {
        MPI_Recv(&x[0][5], 1, coltype, 1, mTag, MPI_COMM_WORLD, &status);
        for (i = 0; i < 4; ++i) {
            printf("Proc %d: my x[%d][5] = %1f\n", ProcRank, i, x[i][5]);
        }
    }
    MPI_Finalize();
}


// Example with structural way of construction
/*
int main(int argc, char** argv) {
    int ProcRank, ProcNum, i, j, mTag = 0;
    
    struct { 
        int i;
        double d[3];
        long l[8];
        char c;
    } MyStruct;

    MyStruct st;
    MPI_Datatype myStructType;
    MPI_Status status;
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    int len[5] = {1, 3, 8, 1, 1};
    MPI_Aint pos[5] = {
        offsetof(MyStruct, i), 
        offsetof(MyStruct, d), 
        offsetof(MyStruct, l), 
        offsetof(MyStruct, c),
        sizeof(MyStruct)
    };
    MPI_Datatype typ[5] = {MPI_INT, MPI_DOUBLE, MPI_LONG, MPI_CHAR, MPI_UB};

    MPI_Type_struct(5, len, pos, typ, &myStructType);
    MPI_Type_commit(&myStructType);
    MPI_Finalize();
}
*/
