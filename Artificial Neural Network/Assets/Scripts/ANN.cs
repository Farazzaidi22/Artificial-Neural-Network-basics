using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN
{
    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden;
    public double alpha; //alpha is how much of prev inputs will effect the current input
    List<Layer> layers = new List<Layer>();

    public ANN(int nI, int nO, int nH, int nPH, double a)
    {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        alpha = a;

        if(numHidden>0)
        {
            layers.Add(new Layer(numNPerHidden, numInputs));

            for(int i = 0; i < numHidden - 1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }

            layers.Add(new Layer(numOutputs, numNPerHidden));
        }

        else
        {
            layers.Add(new Layer(numOutputs, numInputs));
        }
    }

    public List<double> Go(List<double> inputValues, List<double> desiredOutput)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if(inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of inputs must be " + numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);

        for (int i = 0; i < numHidden + 1; i++)
        {
            if(i > 0)
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            for(int j =0; j < layers[i].numNeurons; j++)
            {
                double N = 0;
                layers[i].neurons[j].inputs.Clear();

                for(int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    N += layers[i].neurons[j].weights[k] * inputs[k]; //* ki jagha + hoskta hai
                }

                N -= layers[i].neurons[j].bias;
                layers[i].neurons[j].output = ActivationFunction(N);
                outputs.Add(layers[i].neurons[j].output);
            }
        }

        UpdateWeights(outputs, desiredOutput);

        return outputs;
    }
}
