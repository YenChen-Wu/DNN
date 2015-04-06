all:baseline.cpp
	g++ -std=c++11 baseline.cpp
O3:
	g++ -O3 -std=c++11 baseline.cpp
D:
	g++ -O3 -DDEBUG -g -std=c++11 baseline.cpp
exp1:exp/exp1.cpp
	g++ -O3 -std=c++11 exp/exp1.cpp -o exp/exp1
test:exp/test.cpp
	g++ -O3 -std=c++11 exp/test.cpp -o exp/test
clean:
