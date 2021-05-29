/*root
3.1240
1.2000(Áß±Ù)
-1.0440
*/

#include<iostream>
#include<algorithm>
#include<cmath>
#include<vector>

using namespace std;

double th = 0.0001;
double poly[5] = { 5,-22.4,15.85272,24.161472,-23.4824832 };
double poly_diff[4] = { 5 * 4,-22.4 * 3,15.85272 * 2,24.161472 };
//double poly_ddiff[3] = { 5 * 4 * 3,-22.4 * 3 * 2, 15.85272 * 2 };
vector<double> bisection_root;
vector<double> newton_root;

double fun(double x)
{
	double ret = 0;

	for (int i = 0; i < 5; i++)
	{
		ret += poly[i] * pow(x, 4-i);
	}
	return ret;
}
double fun_(double x)
{
	double ret = 0;
	for (int i = 0; i < 4; i++)
	{
		ret += poly_diff[i] * pow(x, 3 - i);
	}
	return ret;
}
void bisection(double left, double right)
{
	if (fun(left) * fun(right) >= 0)
	{
		cout << left << " " << right << " : ";
		cout << "wrong input\n";
		return;
	}

	double mid=left, mid_=right;
	double err = fabs((mid - mid_)/mid);
	while (err * 100 >= th)
	{
		mid_ = mid;
		mid = (left + right) / 2;
		if (fun(mid) == 0) break;
		else if (fun(mid) * fun(left) < 0) right = mid;
		else left = mid;
		err = fabs((mid - mid_) / mid);
	}
	
	bisection_root.push_back(mid);
}
void newton(double x)
{
	if (fun(x) == 0)
	{
		newton_root.push_back(x);
		return;
	}

	double x_ = x - fun(x) / fun_(x);
	double err = fabs((x_ - x) / x_);
	while (err * 100 >= th)
	{
		x = x_;
		x_ = x - fun(x) / fun_(x);
		if (fun(x_) == 0) break;
		err= fabs((x_ - x) / x_);
	}
	newton_root.push_back(x_);
}

int main()
{
	//Bisection
	bisection(0, 3);
	bisection(-5,0);
	bisection(0, 5);
	cout << "bisection\n";
	for (double i : bisection_root) cout << i << endl;

	//Newton
	newton(-5);
	newton(0.3);
	newton(2.6);
	cout << "newton\n";
	for (double i : newton_root)cout << i << endl;
}