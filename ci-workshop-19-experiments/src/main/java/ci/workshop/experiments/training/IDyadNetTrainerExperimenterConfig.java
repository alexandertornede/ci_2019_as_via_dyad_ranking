package ci.workshop.experiments.training;

import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.experiments.IExperimentSetConfig;

@Sources({ "file:conf/dyadNetTrainerExperimenter.properties" })
public interface IDyadNetTrainerExperimenterConfig extends IExperimentSetConfig {

	@Key("dataset.folder")
	public String getDatasetFolder();
	
	@Key("ranker.folder")
	public String getRankerFolder();
}
