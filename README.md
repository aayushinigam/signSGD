# signSGD: Compressed Optimization for Non-Convex Problems 

### File Structure
- **FashionMNIST_logisticRegression.py**: impelments signSGD on logistic Regression using Fashion MNIST dataset
- **MNIST_signSGD.py**: implements signSGD on neural network using MNIST Handwritten digits as dataset
- **FashionMINST_signSGD.py**: impements signSGD on neural network using MNIST Fashion as dataset
- **hostfile**: contains the maximum number of cores that our program need to be run on

### How to run
- To run FashionMNIST_logisticRegression.py
```
python3 FashionMNIST_logisticRegression.py
```
- To run other two files
```
mpirun --hostfile <hostfile_name> -np <no_of_cores> python3 <filename>.py
```
### Hostfile contents
```
localhost slots=<maximum_no_of_cores_as_required>
```