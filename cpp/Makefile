INCLUDE = -I src

all:
	g++ -std=c++11 $(INCLUDE) src/baseline.cpp
O3:
	g++ -O3 -std=c++11 $(INCLUDE) src/baseline.cpp
D:
	g++ -O3 -DDEBUG -g -std=c++11 $(INCLUDE) src/baseline.cpp
exp1:exp/exp1.cpp
	g++ -O3 -std=c++11 $(INCLUDE) exp/exp1.cpp -o exp/exp1
exp_visual:
	g++ -O3 -std=c++11 $(INCLUDE) exp/exp_visual.cpp -o exp/exp_visual
test:exp/test.cpp
	g++ -O3 -std=c++11 $(INCLUDE) exp/test.cpp -o exp/test
clean:
