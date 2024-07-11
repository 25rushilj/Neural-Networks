import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;

/**
 * Creates an N Layer network which can hold any amount of layers where there is an input layer, an output layers, and
 * n-2 hidden layers. In this class the network is initialized, the memory is allocated and allows the network to train
 * or run without and activation and error calculations are used for end results. This networks also utilizes
 * backpropagation to adjust weights and uses the sigmoid function and tries to reduce the errors and through the test
 * cases put into a truth table allows to see the accuracy of the model. It also prints all parts of the network to
 * allow for understanding as the model continues.
 *
 * Methods:
 * configureNetwork()
 * setTruthTable()
 * allocateMemory()
 * saveWeights()
 * loadWeights()
 * populate()
 * randomize()
 * echo()
 * activationFunction()
 * sigmoid()
 * derivativeActivationFunction()
 * derivativesigmoid()
 * train()
 * runForTraining()
 * calculateAndApplyDeltaWeights()
 * run()
 * runTestCase()
 * setupInputActivation()
 * reportFinalOutputs()
 * main()
 *
 * @author Rushil Jaiswal
 * @version May 3rd, 2024
 *
 */

public class NLayerSubmitted
{
   public double lambda;

   public int numLayers;

   public double lowRand;
   public double highRand;

   public double maxIter;
   public double avgErrorCutoff;
   public int iter;
   public boolean isTraining;

   public int numConnect;

   public double[][] expectedVals;
   public double[][] actualVals;

   public double[][] a;
   public double[][][] weights;

   public int max;

   public double[][] truthTable;

   public double[][] psiValues;
   public double[][] thetas;

   public int [] numActs;
   public int keepAlive;

   public int numTestCases;

   public String truthTableFile;

   public String weightsFile;

   public String WEIGHT_MODE;

   public double avgError = 0.0;

   public static double startTime = 0.0;
   public static double endTime = 0.0;

   static final int INPUT_LAYER = 0;
   public int LAST_HIDDEN;
   public int OUTPUT_LAYER;


   /**
    * Configures the parameters of the network by reading a file fed to it, and it goes through each of the lines and
    * uses a hashmap for key value pairs for efficient retrieval and can assign the values to the variables from the file
    * given. An error is thrown if there is not something on both sides of the equal sign as a variable and a value
    * is needed. Only continues if there are more lines which means there would be # present. Also assigns numActs so that
    * it can hold the configuration and number of activations for the layers.
    *
    */
   public void configureNetwork(String configFile) throws IOException
   {
      File myObj = new File(configFile);
      HashMap<String, String> configValues = new HashMap<>();

      try (Scanner myReader = new Scanner(myObj))
      {
         while (myReader.hasNextLine())
         {
            String line = myReader.nextLine();

            if (line.startsWith("#")) // makes sure there is a line it can continue with
            {
               continue;
            }

            String[] parts = line.split("="); // the parts are split over an equal sign
            if (parts.length != 2) // 2 parts are needed to make sure there is a value and variable
            {
               throw new IOException("Invalid format in config file: " + line);
            }

            configValues.put(parts[0], parts[1]); // uses the left and right of the line with the variable then value
         } // while (myReader.hasNextLine())
      } // try (Scanner myReader = new Scanner(myObj))

      System.out.println(configValues);
      lambda = Double.parseDouble(configValues.get("lambda"));
      isTraining = Boolean.parseBoolean(configValues.get("isTraining"));

      keepAlive = Integer.parseInt(configValues.get("keepAlive"));

      String layerInfo = configValues.get("layerInfo");
      String [] allLayers = layerInfo.split("-");
      numActs = new int[allLayers.length];
      for (int x = 0; x < allLayers.length; x++)
      {
         numActs[x] = Integer.parseInt(allLayers[x]);
      }
      numLayers = numActs.length;

      numConnect = numLayers - 1;
      avgErrorCutoff = Double.parseDouble(configValues.get("avgErrorCutoff"));

      lowRand = Double.parseDouble(configValues.get("lowRand"));
      highRand = Double.parseDouble(configValues.get("highRand"));

      maxIter = Double.parseDouble(configValues.get("maxIter"));

      numTestCases = Integer.parseInt(configValues.get("numTestCases"));
      truthTableFile = configValues.get("truthTableFile");

      weightsFile = configValues.get("weightsFile");

      WEIGHT_MODE = configValues.get("WEIGHT_MODE");

      LAST_HIDDEN = numLayers-2;
      OUTPUT_LAYER = numLayers-1;

      max = Integer.MIN_VALUE;
   } // public void configureNetwork(String configFile) throws IOException

   /**
    * Sets the truth table by using the file passed and reading it the going through and taking the parts for each of the
    * spaces to fill the truth table out. The truth table is adjusted depending on the network and how many inputs
    * and outputs and an exception is thrown if the parts of the truth table don't match or are less than that of the
    * number of inputs and outputs.
    */
   public void setTruthTable() throws IOException
   {
      FileReader fileReader = new FileReader(truthTableFile);
      Scanner scanner = new Scanner(fileReader);

      for (int i=0; i < numTestCases; i++)
      {
         String truthTableLine = scanner.nextLine();
         String[] parts = truthTableLine.split(" "); // the parts are split over spaces

         if (parts.length < numActs[INPUT_LAYER] + numActs[OUTPUT_LAYER])
         {
            throw new IOException("Truth table line does not match config file " + truthTableLine);
         }

         for (int inputsLen = 0; inputsLen < numActs[INPUT_LAYER]; inputsLen++)
         {
            truthTable[i][inputsLen] = Double.parseDouble(parts[inputsLen]);
         }

         for (int outputLen = 0; outputLen < numActs[OUTPUT_LAYER]; outputLen++) //numLayers-1 is num output act
         {
            expectedVals[i][outputLen] = Double.parseDouble(parts[numActs[INPUT_LAYER] + outputLen]);
         }
      } // for (int i=0; i < numTestCases; i++)
   } // public void setTruthTable() throws IOException


   /**
    * Initializes memory for the arrays in the network. This method sets up memory for the truth table, a variable max
    * which holds the maximum size to make sure there is no problem storing anything. Then there is an array for all
    * activations and an array for all weights. Then expected values for the inputs given and actual outputs.
    * If the network is training, memory is also initialized for psi values and the thetas.
    */

   public void allocateMemory()
   {
      truthTable = new double[numTestCases][numTestCases - 1];

      for (int n = 0; n < numLayers; n++) //from input layer
      {
         max = Math.max(max,numActs[n]);
      }

      a = new double[numLayers][max];
      weights = new double[numConnect][max][max];


      expectedVals = new double[numTestCases][numActs[OUTPUT_LAYER]];
      actualVals = new double[numTestCases][numActs[OUTPUT_LAYER]];

      if (isTraining)
      {
         psiValues =  new double[OUTPUT_LAYER][max];
         thetas = new double[OUTPUT_LAYER][max];
      }

   } // public void allocateMemory()


   /**
    * Saves the weights of the neural network to a file. Writes the number of input, hidden, and output activation
    * layers to the file, followed by the weights connecting the input layer to the hidden layer, weights for hidden layers
    * between themselves, and the weights connecting the last hidden layer to the output layer. Each weight is written to
    * a new line in the file. Upon completion, the FileWriter is closed.
    *
    */
   public void saveWeights() throws IOException
   {
      FileWriter writer = new FileWriter(weightsFile);

      for (int n = 0; n <numConnect; n++)
      {
         for (int m = 0; m < max; m++)
         {
            for (int k = 0; k < max; k++)
            {
               writer.write(String.valueOf(weights[n][m][k]) + "\n");

            }
         }
      }
      writer.close();

   }// public void saveWeights() throws IOException


   /**
    * Loads the weights by first reading the weights file. It then goes through and sets the values for the weights from
    * the input to hidden layers, first hidden layer and second hidden layer, then second hidden layer to output layers
    * and then the reader is closed.
    */
   public void loadWeights() throws IOException
   {
      FileReader fileReader = new FileReader(weightsFile);
      Scanner scanner = new Scanner(fileReader);

      for (int n = 0; n < numConnect; n++)
      {
         for (int m = 0; m < max; m++)
         {
            for (int k = 0; k < max; k++)
            {
               weights[n][m][k] = Double.parseDouble(scanner.nextLine());
            }
         }
      }

      scanner.close();
      fileReader.close();
   } // public void loadWeights() throws IOException

   /**
    * This method sets the truth table by calling the setTruthTable() method then if the weight is set to "random"
    * in the config file as weights mode, then the weights will randomly be set using the lowRand and highRand values.
    * Otherwise, whatever values are set in the weights config files will be used.
    */
   public void populate() throws IOException
   {
      setTruthTable();

      if ("random".equals(WEIGHT_MODE))
      {
         for (int n = 0; n <= LAST_HIDDEN; n++)
         {
            for (int m = 0; m < numActs[n]; m++)
            {
               for (int k = 0; k < numActs[n + 1]; k++)
               {
                  weights[n][m][k] = randomize(lowRand, highRand);
               }
            }
         }
      } // if ("random".equals(WEIGHT_MODE))
      else
      {
         loadWeights();
      }
   } //public void populate() throws IOException

   /**
    * It takes parameters for the low and high values and gives random numbers which are assigned to weights in
    * between that range for the truth table if the network is training.
    */
   public double randomize(double low, double high)
   {
      return low + Math.random() * (high - low);
   } // public double randomize(double low, double high)


   /**
    * Echos the parameters for this network by printing out descriptions for the parameters that are given to it along
    * with the network configuration, given based off of the input, hidden, and output activation layer numbers. The
    * parameters include the runtime training parameters if its training or just mentions that the model isn't
    * training. Also says why training has stopped if it has and the keep alive message gap.
    */
   public void echo()
   {
      System.out.println("Network Configuration: ");

      for (int n =0; n < OUTPUT_LAYER; n++)
      {
         System.out.print(numActs[n] + "-");
      }

      System.out.println(numActs[OUTPUT_LAYER]);

      if (isTraining)
      {
         System.out.println("This model is training");
         System.out.println("Runtime Training Parameters: ");

         System.out.println("Random numbers range: " + this.lowRand + " - " + this.highRand);
         System.out.println("Max iterations: " + this.maxIter);
         System.out.println("Error threshold: " + this.avgErrorCutoff);
         System.out.println("Lambda value: " + this.lambda);
         System.out.println("Keep Alive Message Gap " + this.keepAlive);

      } // if (isTraining)
      else
      {
         System.out.println("This model is not training");
      }
   } //public void echo()


   /**
    * Is an activation function wrapper than can call various activation functions but currently is set to call
    * the sigmoid function.
    */
   public double activationFunction(double x)
   {
      return sigmoid(x);
   } // public double activationFunction(double x)

   /**
    * Uses the parameter x and calculates the sigmoid value and returns the result for that.
    */
   public double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   } // public double sigmoid(double x)

   /**
    * A derivative of the activation function that we use for the network training
    */
   public double derivativeActivationFunction(double x)
   {
      return derivativesigmoid(x);
   } // public double derivativeActivationFunction(double x)

   /**
    * Uses the parameter x and calculates the derivative sigmoid value and returns the result for that.
    */
   public double derivativesigmoid(double x)
   {
      double sig = sigmoid(x);
      return sig * (1.0 - sig);
   } // public double derivativesigmoid(double x)


   /**
    * This method goes through the train process for the network then it runs the network. It takes the network
    * configuration and uses the randomly assigned weights for the truth table or the given weights to train the network
    * by first setting the input then calls the runForTraining() method to evaluate then calculates the weights and applies
    * them and then runs each test case. It also calculates all the errors and comes up with a total error as well.
    * It continues doing this until either the amount of iterations of the training surpass the max iterations or
    * the average error for the running of the network is under the average error cutoff. It gives a reason for why
    * training was ended and the training exit information. If the network is not training, then it will go through
    * all the test cases and just run the network.
    */
   public void train()
   {
      boolean done = false;

      while (!done)
      {
         double totalError = 0.0;

         for (int iTruthTable = 0; iTruthTable < numTestCases; iTruthTable++)
         {
            setupInputActivation(iTruthTable); // 1. Set Input

            runForTraining(iTruthTable); // 2. Evaluate

            calculateAndApplyDeltaWeights(); // 3. Weights

            runTestCase(iTruthTable);


            double error = 0.0;
            for (int i = 0; i < numActs[OUTPUT_LAYER]; i++)
            {
               error = 0.5 * (expectedVals[iTruthTable][i] - a[OUTPUT_LAYER][i]) *
                     (expectedVals[iTruthTable][i] - a[OUTPUT_LAYER][i]);  //applying error function
               totalError += error;
            }
         } // for (int iTruthTable = 0; iTruthTable < numTestCases; iTruthTable++)
         avgError = totalError / (double) (numTestCases);
         iter++;                                                   //record iterations before end of cycle

         if (avgError <= avgErrorCutoff)
         {
            done = true;
         }

         if (iter >= maxIter)
         {
            done = true;
         }

         if (keepAlive > 0 && (iter % keepAlive)==0)
         {
            System.out.printf("Iteration %d, Error = %f\n", iter, avgError);
         }
      } // while (!done)
   } // public void train()

   /**
    * Takes an index from the truth table ad goes through the hidden layers and resets the thetas and then goes through
    * input layers to set the thetas to the input activations multiplied by the values of the weights for the input
    * to the hidden layers. The hidden activations are then called with the activation function and the values of the
    * thetas calculated from second hidden layer. It also calculates the hidden to output values.
    */
   private void runForTraining(int iTruthTable)
   {
      for (int n = 0; n < LAST_HIDDEN; n++)
      {
         for (int k = 0; k < numActs[n + 1]; k++)
         {
            thetas[n][k] = 0.0; // reset thetas

            for (int m = 0; m < numActs[n]; m++)
            {
               thetas[n][k] += a[n][m] * weights[n][m][k];
            } // for (int k = 0; k < numInputAct; k++)

            a[n + 1][k] = activationFunction(thetas[n][k]);
         }  // for (int k = 0; k < numActs[n + 1]; k++)
      } // for (int n = 0; n < LAST_HIDDEN; n++)

      int n = LAST_HIDDEN;
      for (int i = 0; i < numActs[OUTPUT_LAYER]; i++)
      {
         thetas[n][i] = 0.0; // resets thetas
         for (int j = 0; j < numActs[n]; j++)
         {
            thetas[n][i] += a[n][j] * weights[n][j][i];
         } // for (int j = 0; j < numHiddenSecondAct; j++)

         a[n+1][i] = activationFunction(thetas[n][i]);

         psiValues[n][i] = (expectedVals[iTruthTable][i] - a[n+1][i]) * derivativeActivationFunction(thetas[n][i]);
      } // for (int i = 0; i < numActs[OUTPUT_LAYER]; i++)
   } // private void runForTraining(int iTruthTable)


   /**
    * Calculates the delta weights by using the index from the truth table going through the hidden layers and output
    * layers and following the design document for setting the values of the weight for the hidden to output layers and
    * input to hidden layers using omegas and psis.
    */
   private void calculateAndApplyDeltaWeights()
   {
      double omega = 0.0;

      for(int n = LAST_HIDDEN; n > 1; n--)
      {
         for (int j = 0; j < numActs[n]; j++)
         {
            omega = 0.0;

            for (int i = 0; i < numActs[n + 1]; i++)
            {
               omega += psiValues[n][i] * weights[n][j][i];

               weights[n][j][i] += -lambda * (-a[n][j] * psiValues[n][i]);

            }

            psiValues[n - 1][j] = omega * derivativeActivationFunction(thetas[n - 1][j]);
         } // for (int j = 0; j < numActs[n]; j++)
      } // for(int n = LAST_HIDDEN; n > 1; n--)

      for (int k = 0; k < numActs[INPUT_LAYER+1]; k++) //first hidden layer
      {
         omega = 0.0;
         int n = INPUT_LAYER+1; // set to first hidden layer

         for (int j = 0; j < numActs[n+1]; j++)
         {
            omega += psiValues[n][j] * weights[n][k][j];

            weights[n][k][j] += lambda * (a[n][k] * psiValues[n][j]);
         }

         psiValues[n-1][k] = omega * derivativeActivationFunction(thetas[n-1][k]);

         n = INPUT_LAYER;
         for (int m = 0; m < numActs[INPUT_LAYER]; m++)
         {
            weights[n][m][k] += lambda * (a[n][m] * psiValues[n][k]);

         }
      } // for (int k = 0; k < numActs[INPUT_LAYER+1]; k++)
   } //private void calculateAndApplyDeltaWeights()

   /**
    * Runs the network with the truth table cases and putting it through the method runTestCase().
    */
   public void run()
   {
      for (int iTruthTable = 0; iTruthTable < numTestCases; iTruthTable++)
      {
         setupInputActivation(iTruthTable);
         runTestCase(iTruthTable);
      } // for (int iTruthTable = 0; iTruthTable < numTestCases; iTruthTable++)
   } // public void run()

   /**
    * With each of the test cases it takes the input activation values from the truth table, then calculates the hidden
    * activations and then the output activations using the activation functions.
    */
   private void runTestCase(int iTruthTable)
   {
      for (int n = INPUT_LAYER; n < OUTPUT_LAYER; n++)
      {
         for (int k = 0; k < numActs[n + 1]; k++)
         {
            double theta = 0;
            for (int m = 0; m < numActs[n]; m++)
            {
               theta += a[n][m] * weights[n][m][k];
            } // for (int m = 0; m < numInputAct; m++)
            a[n + 1][k] = activationFunction(theta);

         } // for (int k = 0; k < numActs[n + 1]; k++)
      } // for (int n = INPUT_LAYER; n < OUTPUT_LAYER; n++)

      for (int i = 0; i < numActs[OUTPUT_LAYER]; i++)
      {
         actualVals[iTruthTable][i] = a[OUTPUT_LAYER][i];
      }
   } // private void runTestCase(int iTruthTable)

   /**
    * Fills in the inputs in the activations array from the truth table read in the config file.
    */
   private void setupInputActivation(int iTruthTable)
   {
      int n = INPUT_LAYER;
      for (int m = 0; m < numActs[INPUT_LAYER]; m++)
      {
         a[n][m] = truthTable[iTruthTable][m];
      }
   } //private void setupInputActivation(int iTruthTable)

   /**
    * Reports all the information from the network at the end from the test cases the final F0 values for each test case
    * the expected values the calculated errors and is formatted to be understood clearly. It also lists the average
    * error and the number of iterations and the time taken at the bottom.
    */
   public void reportFinalOutputs()
   {

      if (avgError <= avgErrorCutoff)
      {
         System.out.println("Training is ending because the average error is less than the cutoff:" + avgError);
      }

      if (iter >= maxIter)
      {
         System.out.println("Training is ending because the iterations have gotten to the max iterations:" + iter);
      }

      System.out.println("\nFinal F0 values for each test case and their expected values:");

      for (int testIndex = 0; testIndex < numTestCases; testIndex++)
      {
         System.out.print("Test case " + (testIndex + 1) + ": Input (");

         for (int inputIndex = 0; inputIndex < numActs[INPUT_LAYER]; inputIndex++)
         {
            System.out.print(" " + String.format("%.0f", truthTable[testIndex][inputIndex]) + " ");
         }

         System.out.print("), Expected Output (");

         for (int i = 0; i < numActs[OUTPUT_LAYER]; i++)
         {
            System.out.print(" " +  (expectedVals[testIndex][i]) + " ");
         }

         System.out.print("), Actual Output (");

         for (int i = 0; i < numActs[OUTPUT_LAYER]; i++)
         {
            System.out.print(" " + String.format("%.18f", actualVals[testIndex][i]) + " ");
         }

         System.out.println(")");

      } // for (int testIndex = 0; testIndex < numTestCases; testIndex++)

      System.out.println("Num of Iterations: " + iter);
      System.out.println("Avg Error: " + avgError);

      System.out.println("Time Taken: " + (endTime-startTime) + "ms");

   } // public void reportFinalOutputs()

   /**
    * Initializes a neural network, configures it based on the provided
    * command-line arguments or default configuration file, and either trains the network or runs it directly
    * depending on the configured mode. If training mode is selected, the network is trained until completion, through
    * the training, the running, and the saving of the weights with runtime training parameters and exit information
    * displayed. If running mode is selected, the network runs using pre-configured weights. Final outputs, including
    * input values, expected outputs, actual outputs, and the number of iterations, are reported. The execution time of
    * the process is also displayed.
    */
   public static void main(String[] args) throws IOException
   {
      NLayer net = new NLayer();

      String configFile = "default.cfg"; //if no file is input

      if (args.length > 0)
      {
         configFile = args[0];
      }

      System.out.println("Using config file " + configFile);
      System.out.println(System.getProperty("user.dir"));
      net.configureNetwork(configFile);
      net.echo();
      net.allocateMemory();
      net.populate();

      startTime = System.currentTimeMillis();

      if (net.isTraining)
      {
         net.train();
         net.run();
         net.saveWeights();
      }
      else
      {
         net.run();
      }
      endTime = System.currentTimeMillis();

      net.reportFinalOutputs();

   } // public static void main(String[] args) throws IOException
} // public class NLayer