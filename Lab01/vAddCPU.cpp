// Elaborar un programa para CPU que sume dos vectores de valores flotantes.
// Mida el tiempo de ejecucion para sumar vectores mayores a 1000 elementos.
#include <bits/stdc++.h>
using namespace std;
using namespace chrono;
vector<float> vAddCPU(vector<float> vec, vector<float> vec2) {
  vector<float> ans;
  for (int i = 0; i < vec.size(); i++) {
    ans.push_back(vec[i] + vec2[i]);
  }
  return ans;
}
int main() {
  int n=100000;
  srand(time(0));
  vector<float> vec, vec2, ans;
  for (int i = 0; i < n; i++) {
    vec.push_back(rand() % 1000);
  }
  srand(time(0) + 1);
  for (int i = 0; i < n; i++) {
    vec2.push_back(rand() % 1000);
  }
  auto t1 = high_resolution_clock::now();
  ans = vAddCPU(vec, vec2);
  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  duration<double, milli> ms_double = t2 - t1;
  cout << "Tiempo: " << ms_double.count() << " ms\n";
  return 0;
}
