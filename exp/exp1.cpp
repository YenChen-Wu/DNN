#include <data-io.h>
#include <util.h>
#include <nnet.h>
//#include <layer.h>
/****************************
EXP1	Backpropagation is not good enough
****************************/
void random_generate(mat& X,mat& Y,nnet* myNN,int nSample);
void saveData(ofstream& out,nnet* nnetA,vector<float>& local_min);

main(){
	int batchSize = 64,epoch = 10;
	float init_var = 0.1;
	// about lr
    float lr = 0.005,decay = 0.9999,momentum = 0.9;
	//reg
    float lamda = 0;

    int hidSize = 128;
	int nSample = 100000;

	cout << "		EXP 1		LIMIT OF BACKPROPAGATION" << endl;
	cout << "nSample = " << nSample << endl;	
	mat X,Y,tX,tY;
	nnet nnetA(69,128,48,nSample,lr,decay,lamda,init_var,momentum);
	nnet nnetB(69,128,48,batchSize,lr,decay,lamda,init_var,momentum);
	random_generate(X,Y,&nnetA,nSample);
	random_generate(tX,tY,&nnetA,nSample);

	// para exp1
	vector<float> local_min;
	float try_times = 5;
	ofstream out;
	out.open("./local_min.csv",fstream::app);

	// nnetB try to fit nnetA
	for (int i = 0;i<try_times;i++){
		cout << "Try " << i+1 << " ";
		nnetB.nn_train(X,Y,tX,tY,epoch);
		local_min.push_back(nnetB.best_Ein);
		nnetB.reset();
	}

	saveData(out,&nnetA,local_min);
}

void random_generate(mat& X,mat& Y,nnet* myNN,int nSample){
	float init_var = 0.1;
	X = randMat(nSample,69,init_var);
	Y = randMat(nSample,48,init_var);
	myNN->getBatch(X,Y,1);
	myNN->feedForward();
	Y = myNN->output;
}

void saveData(ofstream& out,nnet* nnetA,vector<float>& local_min){
	float m = mean(local_min);
	float v = var(local_min);
	out << m << "," << v << "\t";	
	outMatrix(nnetA->w1,out);
	outMatrix(nnetA->w2,out);
	out << "\b" << endl;
}
