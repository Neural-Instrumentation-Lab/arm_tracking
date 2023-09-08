INC = -O3 -I /Applications/MATLAB_R2010a.app/extern/include
LIB = -L/Applications/MATLAB_R2010a.app/bin/maci64
OBJ = -leng -lmx

rbf.o: rbf.h rbf.cpp
	g++ -c rbf.cpp -o rbf.o

cerebellum.o: cerebellum.h cerebellum.cpp rbf.h
	g++ -c cerebellum.cpp -o cerebellum.o

main.o: main.cpp cerebellum.h
	g++ $(INC) -c main.cpp -o main.o

main: main.o cerebellum.o rbf.o
	g++ $(LIB) main.o $(OBJ) cerebellum.o rbf.o -o main

test: test.cpp
	g++ $(INC) $(LIB) $(OBJ)  test.cpp -o test

plant: testPlant.cpp plant.cpp plant.h
	g++ testplant.cpp plant.cpp -o testplant
