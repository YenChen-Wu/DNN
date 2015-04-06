#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <ctime>

using namespace std;
typedef vector< vector<float> > mat;

mat readFile(const char* filename){
	time_t tstart, tend; 
	tstart = time(0);
	cerr << "Reading " << filename << "..."; 

    mat X;
    string line;
    ifstream infile(filename);
    float n;

    if (infile.is_open()){
        while ( getline (infile,line) ){
            //if (line[0]=='t'){
            if (line[0] > 58){
                continue;
            }
            stringstream stream(line);
            vector<float> x;
            while(stream >> n){
                x.push_back(n);
            }
            X.push_back(x);
        }
        infile.close();
    }
    else cout << "Unable to open file" << endl;

	tend = time(0);
	cout << "	Total " << X.size() << " frames	";
	cout << "It took "<< difftime(tend, tstart) <<" second(s)."<< endl;

    return X;
}

vector<int> readLabel(const char* filename){
	time_t tstart, tend;
    tstart = time(0);
    cout << "Reading " << filename << "...";

    vector<int> Y;
    string line, seq_name;
    ifstream infile(filename);
    if (infile.is_open()){
        while ( getline (infile,line) ){
            stringstream stream(line);
            stream >> seq_name;
            int n;
            while(stream >> n){
                Y.push_back(n);
            }
        }
        infile.close();
    }
    else cout << "Unable to open file" << endl;

	tend = time(0);
	cout << "	It took "<< difftime(tend, tstart) <<" second(s)."<< endl;
    return Y;
}
