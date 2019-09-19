package ci.workshop.experiments.training;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import org.aeonbits.owner.ConfigFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.IDatabaseConfig;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.core.exception.TrainingException;
import ai.libs.jaicore.ml.dyadranking.algorithm.IPLNetDyadRankerConfiguration;
import ai.libs.jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import ci.workshop.experiments.datasetgen.DatasetGenerator;

public class DyadNetTrainerExperimenter implements IExperimentSetEvaluator {

	private Logger logger = LoggerFactory.getLogger(DyadNetTrainerExperimenter.class);
	private static IDyadNetTrainerExperimenterConfig experimenterConfig = ConfigFactory
			.create(IDyadNetTrainerExperimenterConfig.class);

	@Override
	public void evaluate(ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor)
			throws ExperimentEvaluationFailedException, InterruptedException {
		try {
			Map<String, String> keyFields = experimentEntry.getExperiment().getValuesOfKeyFields();
			int splitNum = 
					Integer.parseInt(keyFields.get("splitNum"));
			int numTrainingPairs =  Integer.parseInt(keyFields.get("numTrainingPairs"));

			// read training data
			String location = experimenterConfig.getDatasetFolder() + "/" + DatasetGenerator.getFileName(splitNum,numTrainingPairs);
			logger.info("Reading training data from location {}", location);
			DyadRankingDataset dataset = new DyadRankingDataset();
			dataset.deserialize(new FileInputStream(new File(location)));

			// load plnet conf
			IPLNetDyadRankerConfiguration config;
			Properties properties = new Properties();
			properties.load(new FileInputStream(new File("conf/plnet.properties")));
			config = ConfigFactory.create(IPLNetDyadRankerConfiguration.class, properties);

			// train net
			logger.info("Training PLNet");
			PLNetDyadRanker ranker = new PLNetDyadRanker(config);
			ranker.train(dataset);

			// save net
			new File(experimenterConfig.getRankerFolder()).mkdirs();
			String rankerLocation = experimenterConfig.getRankerFolder() + "/" + String.format("plnet_%d_%d", splitNum, numTrainingPairs);
			logger.info("Saving PLNet to {}.zip", rankerLocation);
			ranker.saveModelToFile(rankerLocation);
			
			Map<String, Object> results = new HashMap<>();
			results.put("done", true);
			processor.processResults(results);			

		} catch (IOException | TrainingException e) {
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	public static void main(String[] args)
			throws ExperimentDBInteractionFailedException, InterruptedException, IOException {
		Properties properties = new Properties();
		properties.load(new FileInputStream(new File("conf/db.properties")));
		IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class, properties);
		IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(dbConfig);
		ExperimentRunner runner = new ExperimentRunner(experimenterConfig, new DyadNetTrainerExperimenter(), dbHandle);
		System.out.println("conducting experiment");
		runner.randomlyConductExperiments(1);
		System.out.println("experiment conducted");
	}

}
