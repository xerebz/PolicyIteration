import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.domain.singleagent.graphdefined.GraphDefinedDomain;
import burlap.oomdp.auxiliary.common.NullTermination;
import burlap.oomdp.core.*;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

import java.util.Arrays;
import java.util.Random;

public class MDPLife {
    GraphDefinedDomain gdd;
    Domain domain;
    State initState;
    RewardFunction rf;
    TerminalFunction tf;
    DiscreteStateHashFactory hashFactory;
    int numStates;

    public MDPLife(double[][][] matrix) {
        this.numStates = 30;
        this.gdd = new GraphDefinedDomain(numStates);

        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < 30; i++) {
                for (int j = 0; j < 30; j++) {
                    // state, action, next state, probability
                    this.gdd.setTransition(i, k, j, matrix[k][i][j]);
                }
            }
        }

        this.domain = this.gdd.generateDomain();
        this.initState = GraphDefinedDomain.getState(domain, 0);
        this.rf = new CustomRF(matrix);
        this.tf = new NullTermination();
        this.hashFactory = new DiscreteStateHashFactory();
    }

    public static class CustomRF implements RewardFunction {
        double[][][] matrix;

        public CustomRF(double[][][] matrix) {
            this.matrix = matrix;
        }

        @Override
        public double reward(State s, GroundedAction a, State sprime) {
            int i = GraphDefinedDomain.getNodeId(s);
            int j = GraphDefinedDomain.getNodeId(sprime);
            if (a.toString().equals("action0")) {
                return this.matrix[2][i][j];
            } else if (a.toString().equals("action1")) {
                return this.matrix[3][i][j];
            } else {
                throw new RuntimeException("Unknown action: " + a.toString());
            }
        }
    }

    /*
    private ValueIteration computeValue(double gamma) {
        double maxDelta = 0.0001;
        int maxIterations = 1000;
        ValueIteration vi = new ValueIteration(this.domain, this.rf, this.tf, gamma,
                this.hashFactory, maxDelta, maxIterations);
        vi.planFromState(this.initState);
        return vi;
    }
    */

    private PolicyIteration computePolicy(double gamma) {
        double maxDelta = 0.0001;
        int maxEvaluationIterations = 1000;
        int maxPolicyIterations = 1000;
        PolicyIteration pi = new PolicyIteration(this.domain, this.rf, this.tf, gamma,
                this.hashFactory, maxDelta, maxEvaluationIterations, maxPolicyIterations);
        pi.planFromState(this.initState);
        return pi;
    }

    public String runPI(double gamma) {
        PolicyIteration pi = computePolicy(gamma);

        double[] P = new double[this.numStates];
        for (int i = 0; i < this.numStates; i++) {
            State s = GraphDefinedDomain.getState(this.domain, i);
            P[i] = pi.value(s);
        }
        return Arrays.toString(P);
    }

    public static void main(String[] args) {

        double[][][] matrix = new double[4][30][30];
        // transition probabilities
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < 30; i++) {
                for (int j = 0; j < 30; j++) {
                    // equally likely to end up in any state
                    matrix[k][i][j] = 1./30;
                }
            }
        }
        // rewards
        Random r = new Random();
        for (int k = 2; k < 4; k++) {
            for (int i = 0; i < 30; i++) {
                for (int j = 0; j < 30; j++) {
                    // random reward for each transition
                    matrix[k][i][j] = r.nextDouble();
                }
            }
        }

        // System.out.println(Arrays.deepToString(matrix));

        MDPLife mdp = new MDPLife(matrix);

        // discount factor of 3/4
        double gamma = 0.75;
        System.out.println("State values: " + mdp.runPI(gamma));

    }
}