#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;
typedef vector< vector<float> > mat;


//???
void normalize(mat& A){
	for (size_t i = 0; i<A.size(); ++i){
		float ave = 0;
		for (size_t j = 0; j<A[0].size(); ++j){
			ave += A[i][j];
		}
		ave /= A[0].size();
		for (size_t j = 0; j<A[0].size(); ++j){
			A[i][j] -= ave;
		}
		
		float var = 0;
		for (size_t j = 0; j<A[0].size(); ++j){
			var += pow(A[i][j],2);
		}
		var /= A[0].size();

		for (size_t j = 0; j<A[0].size(); ++j){
			A[i][j] /= sqrt(var);
		}


	}


}
mat& softmax(mat& A){
    for (size_t i = 0; i<A.size(); ++i){
		float sum = 0;
		// find max
		float max = *max_element(A[i].begin(),A[i].end());
        for (size_t j = 0; j<A[0].size(); ++j)
			A[i][j]	-= max;
        for (size_t j = 0; j<A[0].size(); ++j)
			sum += exp(A[i][j]);
        for (size_t j = 0; j<A[0].size(); ++j)
            A[i][j] = exp(A[i][j])/sum;
	}
	return A;
}

mat& sigmoid(mat& A){
    for (size_t i = 0; i<A.size(); ++i)
        for (size_t j = 0; j<A[0].size(); ++j)
            A[i][j] = 1/(1+exp(-A[i][j]));
	return A;
}

bool checkIsProb(mat& A){
    for (size_t i = 0; i<A.size(); ++i)
        for (size_t j = 0; j<A[0].size(); ++j)
            if(A[i][j]<0||A[i][j]>1)return false;
	return true;
}

void printMatSize(const mat& A){
	cout << A.size() << "x" << A[0].size() << endl;
}

void printMatrix(const mat& A){
    for (size_t i = 0; i<A.size(); ++i){
        for (size_t j = 0; j<A[0].size(); ++j)
            cout << A[i][j] << " ";
        cout << endl;
    }
	printMatSize(A);
}

void outMatrix(const mat& A, ofstream& out){
    for (size_t i = 0; i<A.size(); ++i){
        for (size_t j = 0; j<A[0].size(); ++j)
            out << A[i][j] << ",";
    }
}

void checkSum(const mat& A){
    for (size_t i = 0; i<A.size(); ++i){
		float sum = 0;
        for (size_t j = 0; j<A[0].size(); ++j)
			sum += A[i][j];
		cout << sum << endl;
	}
	
}

mat matrixMul(const mat& A,const mat& B){
    int M = A.size();
    int N = B.size();
    int L = B[0].size();
    mat output(M,vector<float>(L));

    for (size_t i = 0; i<M; ++i)
        for (size_t j = 0; j<N; ++j)
            for (size_t k = 0; k<L; ++k)
                output[i][k] += A[i][j]*B[j][k];
    return output;
}

mat randMat(int M,int N,float var){
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	normal_distribution<float> distribution(0,var);

	mat A(M,vector<float>(N));

    for (size_t i = 0; i<M; ++i)
        for (size_t j = 0; j<N; ++j)
			A[i][j] = distribution(generator);
	return A;
}

mat zeroMat(int M,int N){
	mat A(M,vector<float>(N));
    for (size_t i = 0; i<M; ++i)
        for (size_t j = 0; j<N; ++j)
			A[i][j] = 0;
	return A;
}

int get01Error(const mat& A,const mat& B){
	int err = 0;
	for(size_t i = 0;i<A.size();i++){
 		int id1 = max_element(A[i].begin(),A[i].end())-A[i].begin();
 		int id2 = max_element(B[i].begin(),B[i].end())-B[i].begin();
		if(id1 != id2) err++;
	}
	return err;
}

float getAcc(const mat& A,const mat& B){
	int acc = 0;
	for(size_t i = 0;i<A.size();i++){
 		int id1 = max_element(A[i].begin(),A[i].end())-A[i].begin();
 		int id2 = max_element(B[i].begin(),B[i].end())-B[i].begin();
		if(id1 == id2) acc++;
	}
	return (float)acc/A.size();
}

float getError(const mat& target,const mat& output){
	float err = 0;
    for(size_t i = 0; i<target.size();i++ ){
        for(size_t j = 0; j<target[0].size();j++ ){
            // L2
            //err += pow((target[i][j] - output[i][j]),2);

            // Cross Entropy
            if (output[i][j] == 0)err += target[i][j] * 10000;
            else	err += -target[i][j] * log(output[i][j]);
        }
    }
    return err;
}

mat lab2mat(const vector<int>& label){
    mat Y;
    for(size_t i = 0;i<label.size();i++){
        vector<float> tmp(48,0);
        tmp[label[i]] = 1;
        Y.push_back(tmp);
    }
    return Y;
}

float L2(vector<float> a,vector<float> b){
	float sum = 0;
	for(size_t i = 0;i<a.size();i++){
		sum += pow((a[i]-b[i]),2);
	}
	return sum;
}

bool checkNaN(const mat& A){
    for (size_t i = 0; i<A.size(); ++i)
        for (size_t j = 0; j<A[0].size(); ++j)
			if(A[i][j]!=A[i][j]) return true;
	return false;
}

bool checkZero(const mat& A){
    for (size_t i = 0; i<A.size(); ++i)
        for (size_t j = 0; j<A[0].size(); ++j)
			if(A[i][j]==0) return true;
	return false;
}

mat addBias(const mat& A){
	mat B(A);
	for (size_t i = 0;i<B.size();i++){
		B[i].push_back(1);
	}
	return B;
}

void checkWeight(const mat& A){
	vector<float> max,min;
	for(size_t i = 0;i<A.size();i++){
 		int id_max = max_element(A[i].begin(),A[i].end())-A[i].begin();
 		int id_min = min_element(A[i].begin(),A[i].end())-A[i].begin();
		max.push_back(A[i][id_max]);
		min.push_back(A[i][id_min]);
	}
	cout << "Max Weight " << max[max_element(max.begin(),max.end())-max.begin()] << "	";
	cout << "Min Weight " << min[min_element(min.begin(),min.end())-min.begin()] << endl;
}

void shrinkWeight(mat& A,float rate){
    for (size_t i = 0; i<A.size(); ++i)
        for (size_t j = 0; j<A[0].size(); ++j)
			A[i][j]*=rate;
}

void shuffle(mat& A,vector<int> v){
	for (size_t i = 0;i<A.size(); ++i){
		swap(A[i],A[v[i]]);
	}
}

vector<int> randvec(int max){
	vector<int> v;
	for(size_t i = 0;i<max; ++i)
		v.push_back(rand()%max);
	return v;
}

float mean(vector<float>& v){
	float ans = 0;
	for(size_t i = 0;i<v.size();i++)
		ans += v[i];
	return ans/v.size();
}
float var(vector<float>& v){
	float ans = 0,m = mean(v);
	for(size_t i = 0;i<v.size();i++)
		ans += pow(v[i]-m,2);
	return ans/v.size();
}

void randShuffle(mat& X,mat& Y){
	vector<int> v = randvec(X.size());
	shuffle (X,v);
	shuffle (Y,v);
}
