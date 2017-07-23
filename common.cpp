#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
using namespace std;
using namespace Eigen;

typedef struct{
  int h;
  int w;
}Point;

class Image_mat{
private:
  const int MAX = 10000;

public:
  int W, H;
  MatrixXd data;
  int label;

  bool readdata(string filename){
    string str;
    ifstream fin(filename);
    if(fin){
      fin >> str;
      fin.ignore();
      if(str != "P2") cout << "file format error" << endl;
      while(getline(fin, str), str[0]=='#'); // コメント読み捨て
      sscanf(str.data(), "%d %d", &W, &H);
      data.resize(W*H, 1);
      fin >> str; //255
      int i=0;
      while(fin >> str){
        data(i) = stod(str)/255;
        i++;
      }

      return true;
    }
    else return false;
  }

};

void print_time(chrono::system_clock::time_point start){
  auto end = chrono::system_clock::now();     
  auto dur = end - start;       
  auto msec = chrono::duration_cast<chrono::milliseconds>(dur).count();
  cout << msec << " milli sec" << endl;
}


MatrixXd softmax(MatrixXd X){
  X.array() -= X.maxCoeff();
  return exp(X.array()) / exp(X.array()).sum();
}

class Affine_layer{
public:
  MatrixXd W, b, dW, db, X;

  Affine_layer(){}
  Affine_layer(MatrixXd W, MatrixXd b){
    this->W = W;
    this->b = b;
  }
  MatrixXd forward(MatrixXd X){
    this->X = X;
    return W*X + b;
  }
  MatrixXd backward(MatrixXd dout){
    MatrixXd dx = W.transpose()*dout;
    dW = dout*X.transpose();
    db = dout;
    return dx;
  }
  void update(double l_rate){
    W -= l_rate*dW;
    b -= l_rate*db;
  }
};

class Relu{
public:
  Matrix<bool,Dynamic, Dynamic> mask;

  Relu(){}
  MatrixXd forward(MatrixXd X){
    mask = X.array()<0;
    for(int i=0; i<X.rows(); i++)
      for(int j=0; j<X.cols(); j++)
        if(mask(i, j))
          X(i, j)=0;
    return X;
  }
  MatrixXd backward(MatrixXd dout){
    for(int i=0; i<dout.rows(); i++)
      for(int j=0; j<dout.cols(); j++)
        if(mask(i, j))
          dout(i, j)=0;
    return dout;
  }
  void update(double l_rate){}
};

class Softmaswithloss{
public:
  MatrixXd Y;
  int label;

  Softmaswithloss(){}
  double forward(MatrixXd X, int label){
    Y = softmax(X);
    this->label = label;
    double loss = -log(Y(label)+exp(-7)); // inf回避
    return loss;
  }
  MatrixXd backward(){
    Y(label) -= 1;
    return Y;
  }
  void update(double l_rate){}
};

