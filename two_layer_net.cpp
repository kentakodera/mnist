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
  vector<vector<int>> wrong;
  double l;

  Two_layer_net(){}
  Two_layer_net(int input, int hidden, int output, double l_rate){
    layer1.W = MatrixXd::Random(hidden, input).array();
    layer1.b = MatrixXd::Zero(hidden, 1);
    layer3.W = MatrixXd::Random(output, hidden).array();
    layer3.b = MatrixXd::Zero(output, 1);
    this->l_rate = l_rate;
    result.resize(11);
    wrong.resize(10);
    for(int i=0; i<11; i++){
      result[i].assign(11,0);
      wrong[i].clear();
    }
    
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
    if(label != ans)
      wrong[ans].push_back(test.filenumber);
    
  }

  void save_weights(string filename){
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

  void print_result(){
    ofstream ofs;
    ofs.open("accuracy"+to_string(l_rate)+".csv", ios::app);
    for(int i=0; i<10; i++){
      for(int j=0; j<11; j++){
        printf("%5d",result[i][j]);
      }
      double accuracy = (double)result[i][i] / result[i][10];
      cout << "   accuracy:"<< accuracy << endl;
      ofs << accuracy << ","; 
    }
    ofs << endl;
  }

  void save_wrong(string filename){
    ofstream ofs;
    ofs.open(filename, ios::app);
    for(int i=0; i<10; i++){
      ofs << i << ": ,";
      for(int j=0; j<wrong[i].size(); j++){
        ofs << wrong[i][j] << ", ";
      }
      ofs << endl;
    }
    ofs << endl;
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
        train_data[j].filenumber = j;
      }
    }
  }
  cout << "finished reading train data" << endl;  print_time(start);

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
        test_data[j].filenumber = j;
      }
    }
  }
  cout << "finished reading test data " << endl;  print_time(start);

  // parameters
  int input = 784;
  int hidden = 100;
  int output = 10;
  int repeat = 1;
  double l_rate = 0.005;

  Two_layer_net network(input, hidden, output, l_rate);
  //network.load("weight.txt");

  for(int i=0; i<=repeat; i++){

    if(i != 0){ 
      for(int j=0; j<TRAIN; j++)
        network.train(train_data[j]);

      cout << "trained" << i << endl;  print_time(start);
    }

    for(int i=0; i<10; i++){
      network.result[i].assign(11,0);
      network.wrong[i].clear();
    }

    for(int j=0; j<TEST; j++)
      network.test(test_data[j]);

    cout << "tested" << i << endl;  print_time(start);
    network.print_result();

    if(i%(repeat/2)==0)
      network.save_weights("weights/weight"+ to_string(i) +".txt");

    if(i!=0 && (i%10==0 || i==1))
      network.save_wrong("wrong"+to_string(l_rate)+".txt");
    
  }

  return 0;
}



