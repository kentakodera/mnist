#include "common.cpp"

class Two_layer_net{
private:
  //map<string, MatrixXd> param, grad;
  Affine_layer layer1;
  Relu layer2;
  Affine_layer layer3;
  Softmaswithloss lastlayer;
  double l_rate;

public:
  vector<vector<int>> result;
  double l;

  Two_layer_net(){}
  Two_layer_net(int input, int hidden, int output, double l_rate){
    layer1.W = MatrixXd::Random(hidden, input).array();
    layer1.b = MatrixXd::Zero(hidden, 1);
    layer3.W = MatrixXd::Random(output, hidden).array();
    layer3.b = MatrixXd::Zero(output, 1);
    this->l_rate = l_rate;
    result.resize(11);
    for(int i=0; i<11; i++)
      result[i].assign(11,0);
  }

  MatrixXd predict(MatrixXd X){
    X = layer1.forward(X);
    X = layer2.forward(X);
    X = layer3.forward(X);
    return X;
  }

  double loss(MatrixXd X, int label){
    MatrixXd Y = predict(X);
    return lastlayer.forward(Y, label);
  }

  void gradient(){
    MatrixXd dout;
    dout = lastlayer.backward();
    dout = layer3.backward(dout);
    dout = layer2.backward(dout);
    dout = layer1.backward(dout);
  }

  void update(){
    layer1.update(l_rate);
    layer3.update(l_rate);
  }

  void train(Image_mat train){
    MatrixXd X = train.data;
    int label = train.label;

    l = loss(X, label);
    gradient();
    update();
  }

  void test(Image_mat test){
    MatrixXd X = test.data;
    int label = test.label;
    X = predict(X);
    int ans;
    for(int i=0; i<X.rows(); i++)
      for(int j=0; j<X.cols(); j++) // j=1
        if(X.maxCoeff() == X(i, j))
          ans = i; // 最大要素が複数あれば後の方が採用される

    result[label][ans]++;
    result[label][10]++;
  }

  void save(string filename){
    ofstream ofs;
    ofs.open(filename, ios::out);
    ofs << layer1.W.rows() << " " << layer1.W.cols() << endl;
    ofs << layer1.W << endl;
    ofs << layer1.b.rows() << " " << layer1.b.cols() << endl;
    ofs << layer1.b << endl;
    ofs << layer3.W.rows() << " " << layer3.W.cols() << endl;
    ofs << layer3.W << endl;
    ofs << layer3.b.rows() << " " << layer3.b.cols() << endl;
    ofs << layer3.b << endl;
  }

  void load(string filename){
    string str;
    int r, c;
    ifstream fin(filename);
    if(fin){
      fin >> r >> c;
      layer1.W.resize(r, c);
      for(int i=0; i<r; i++)
        for(int j=0; j<c; j++)
          fin >> layer1.W(i, j);

      fin >> r >> c;
      layer1.b.resize(r, c);
      for(int i=0; i<r; i++)
        for(int j=0; j<c; j++)
          fin >> layer1.b(i, j);

      fin >> r >> c;
      layer3.W.resize(r, c);
      for(int i=0; i<r; i++)
        for(int j=0; j<c; j++)
          fin >> layer3.W(i, j);

      fin >> r >> c;
      layer3.b.resize(r, c);
      for(int i=0; i<r; i++)
        for(int j=0; j<c; j++)
          fin >> layer3.b(i, j);
    }
  }

};

int main(){
  srand((unsigned int) time(0));
  auto start = chrono::system_clock::now();

  const int TRAIN = 60000;
  vector<Image_mat> train_data(TRAIN);
  for(int i=0; i<=9; i++){
    for(int j=0; j<TRAIN; j++){
      string filename;
      stringstream ss;
      ss << setfill('0') << setw(5) << right << to_string(j+1) << ".pgm";
      ss >> filename;
      if(train_data[j].readdata("images/train_img_pgm/"+to_string(i)+"/img"+filename)){
        train_data[j].label = i;
      }
    }
  }
  cout << "finished reading train data" << endl;   
  print_time(start);

  const int TEST = 10000;
  vector<Image_mat> test_data(TEST);
  for(int i=0; i<=9; i++){
    for(int j=0; j<TEST; j++){
      string filename;
      stringstream ss;
      ss << setfill('0') << setw(5) << right << to_string(j+1) << ".pgm";
      ss >> filename;
      if(test_data[j].readdata("images/test_img_pgm/"+to_string(i)+"/img"+filename)){
        test_data[j].label = i;
      }
    }
  }
  cout << "finished test train data " << endl;
  print_time(start);

  // parameters
  int input = 784;
  int hidden = 100;
  int output = 10;
  double l_rate = 0.01;
  int repeat = 1;

  Two_layer_net network(input, hidden, output, l_rate);
  //network.load("weight.txt");

  for(int i=0; i<repeat+1; i++){

    for(int i=0; i<11; i++)
      network.result[i].assign(11,0);

    for(int j=0; j<TEST; j++)
      network.test(test_data[j]);

    for(int j=0; j<10; j++){
      for(int k=0; k<11; k++){
        printf("%5d",network.result[j][k]);
      }
      cout << "   accuracy:"<< (double)network.result[j][j] / network.result[j][10] << endl;
    }

    network.save("weight"+ to_string(i)+".txt");

    if(i != repeat) 
      for(int j=0; j<TRAIN; j++)
        network.train(train_data[j]);
    
    print_time(start);
  }

  return 0;
}



