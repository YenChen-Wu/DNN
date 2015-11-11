class layer {
public:
	layer(int batch,int in,int out,float var){
		w = randMat(in+1,out,var);
		momentum = zeroMat(in+1,out);
		
	}
	// all is sigmoid

	string type;	//
	mat error;
	mat input;
	layer* _next;
	layer* _pre;
	mat w;
	mat momentum;

	void feedForward(){
		//_next->input = sigmoid(matrixMul(addBias(input),w));
	}

	void backPropagate(
		//_next->error = matrixMul(_pre->error,w)*derivative();
	);

	float derivative(){
		
	}

};
