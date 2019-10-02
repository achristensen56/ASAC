tic

% Biafra Ahanonu
% started 2019.02.11
% Automated classification of CELLMax calcium imaging data outputs

clear summaryStats;

% Place input directories to analyze here
% inputMoviePaths = {...
% 'H:\Biafra_CLEAN_training_data\Adam0_20190228',
% 'H:\Biafra_CLEAN_training_data\Adam0_20190302',
% 'H:\Biafra_CLEAN_training_data\Adam0_20190303',
% 'H:\Biafra_CLEAN_training_data\Adam0_20190310',
% 'H:\Biafra_CLEAN_training_data\Adam0_20190315'
% };


dum_ = dir('H:\Biafra_CLEAN_training_data\*_*');
inputMoviePaths = {};
for i = 1:length(dum_)
    inputMoviePaths{i} = ['H:\Biafra_CLEAN_training_data\' dum_(i).name];
    
end

%inputMoviePaths{end+1} = ['G:\adam0_20190702\'];
%inputMoviePaths{end+1} = ['G:\adam0_20190704\'];
inputMoviePaths{end+1} = ['G:\adam0_20190716\'];
% Put in non-classifier data

% Save classification summary to a private folder here
dataSaveOutputPath = 'private\data\cell_classification';
if ~exist(dataSaveOutputPath,'dir');mkdir(dataSaveOutputPath);end

% =======
% Int: which session(s) to use for training
trainSessionNo = 1:22;
% Int: which session(s) to only classify (e.g. they have no manual classifications). If empty, do normal testing.
classifyOnlySessionNo = [23];
% Binary: 1 = load the data, 0 = do not load the data (turn off if already loaded into workspace and just testing classifier).
loadData = 1;
% Binary: 1 = pre-compute image features for classifier, 0 = use existing pre-computed features (turn off if features already loaded in workspace).
preComputeSwitch = 1;
% Binary: 1 = save classifier output to folder along with binary decisions
saveClassiferOutput = 1;
% Char: Suffix for classifier output MAT-file name
saveNameClassifierStruct = 'classifierOutputChoices';
% =======
% Int: Cost structure, the rows correspond to the true class and the columns correspond to the predicted class
% [TN, FP;
%  FN, TP]
% Increase FP cost to boost sensitivity
% Increase FN cost to boost specificity
costStructure = [0,3;130,0];

% Cell: Cell array of strings with list of movie features to add.
movieFeatures = {'imageMovieCorr','imageMovieCorrThres'};
% Float: Threshold for cell vs. non-cell, value between 0 and 1.
classificationsThreshold = 0.5;
% ==========
% Dataset options
% Name of signal extraction method
signalExtractionMethod = 'EXTRACT';
% Char: regular expression for movies to use, e.g. dfof or dfstd
inputMoviePathRegexp = {'_dff','_dfof','_dfstd'};
% Char: regular expression for raw CELLMax outputs
rawExtractionRegexp = {'_extractAnalysis','rec_extract'};
% rawExtractionRegexp = 'rec_extract';

% Char: regular expression for sorted CELLMax outputs
classificationRegexp = {'_extractAnalysisSorted','decision'};
% classificationRegexp = 'decision';

% Char: HDF5 dataset name
% h5DatasetName = '/1';
h5DatasetName = '/Data/Images';
% Int: 1 = read movies from disk when computing classification features
readMovieChunks = 1;
% Int vector: whether to use a subset of CELLMax signals when classifying
filterList = [];
% ID for this run
runID = datestr(now,'yyyymmdd_HHMM','local');
% ==========
% warm up tic-toc
for ii = 1:1e4
	tic;t=toc;
end
% ==========
startTime = tic;
nMovies = length(inputMoviePaths);

% Load the cell images, activity traces, and movies (or movie paths)
if loadData==1
	inputImages = {};
	inputSignals = {};
	inputTargets = {};
	inputMovie = {};
	for movieNo = 1:nMovies
		display(repmat('-',1,7))
		fprintf('Loading movie %d/%d: %s\n',movieNo,nMovies,inputMoviePaths{movieNo})

		% Load movies or add movie paths when reading from HDF5
		inputMoviePath = getFileList(inputMoviePaths{movieNo},inputMoviePathRegexp);
		inputMoviePath = inputMoviePath{1};
		fprintf('Using movie file: %s\n',inputMoviePath)
		if readMovieChunks==0
			inputMovie{movieNo} = loadMovieList(inputMoviePath,'inputDatasetName',h5DatasetName,'largeMovieLoad',1);
		else
			inputMovie{movieNo} = inputMoviePath;
		end

		% Load signal extraction outputs
		rawExtractionPath = getFileList(inputMoviePaths{movieNo},rawExtractionRegexp);
		rawExtractionPath = rawExtractionPath(~cellfun(@(x) ~isempty(regexp(x,'Sorted')),rawExtractionPath));
		rawExtractionPath = rawExtractionPath{1};
		fprintf('Loading: %s\n',rawExtractionPath)
        
        %%ADAM CHANGED
		extractAnalysisOutput = load(rawExtractionPath);
		% inputImages{movieNo} = emAnalysisOutput.cellImages;
		% inputSignals{movieNo} = emAnalysisOutput.scaledProbabilityAlt;

		% inputImages{movieNo} = output.spatial_weights;
		% inputSignals{movieNo} = permute(output.temporal_weights,[2 1]);

		inputImages{movieNo} = extractAnalysisOutput.filters;
		inputSignals{movieNo} = extractAnalysisOutput.traces;
        if size(extractAnalysisOutput.traces,1)>size(extractAnalysisOutput.traces,2)
            inputSignals{movieNo} = permute(extractAnalysisOutput.traces,[2 1]);
        end

		inputSignals{movieNo}(isnan(inputSignals{movieNo})) = 0;
        clear emAnalysisOutput

		% Load classifications
		classificationPath = getFileList(inputMoviePaths{movieNo},classificationRegexp);
		if isempty(classificationPath)
			disp('No manual classifications! Make sure this folder is set as a classify only session (see classifyOnlySessionNo variable)')
			inputTargets{movieNo} = [];
		else
			classificationPath = classificationPath{1};
			fprintf('Loading: %s\n',classificationPath)
			load(classificationPath)
			% inputTargets{movieNo} = logical(validCELLMax);
            if exist('validEXTRACT','var')~=0
    			inputTargets{movieNo} = logical(validEXTRACT);
            elseif exist('decision','var')~=0
    			inputTargets{movieNo} = logical(decision); 
            end
        end
        clear decision validEXTRACT

		if isempty(filterList)
		else
			inputImages = inputImages(:,:,filterList);
			inputSignals = inputSignals(filterList,:);
			inputTargets = inputTargets(filterList);
		end
	end
end
end0 = toc(startTime);

% profile on
% Pre-compute features for classifier
if preComputeSwitch==1
	preComputedFeatures = {};
	for movieNo = 1:nMovies
		% To train classifiers on a set of images/signals
		[classifyStruct] = classifySignals(inputImages(movieNo),inputSignals(movieNo),...
			'classifierType','all',...
			'trainingOrClassify','training',...
			'inputMovieList',inputMovie(movieNo),...
			'inputTargets',inputTargets(movieNo),...
			'readMovieChunks',readMovieChunks,...
			'onlyComputeFeatures',1,...
			'oversampleTrainingData',1,...
			'plotTargetVsFeatures',1,...
			'movieFeatures',movieFeatures,...
			'inputDatasetName',h5DatasetName);
		preComputedFeatures{movieNo} = classifyStruct.inputFeatures;
	end
	end1 = toc(startTime);
end

% Classify directories without existing manual classifications
if ~isempty(classifyOnlySessionNo)
	% Create classifier both normal, shuffled, and setting targets to various extreme cases
	analysisState = {'normal'};
	nStates = length(analysisState);
	for analysisNo = 1:nStates
		analysisStateStr = analysisState{analysisNo};
		plotTargetVsFeatures = 0;

		trainTargets = inputTargets(trainSessionNo);
		plotTargetVsFeatures = 1;
		trainFxn = @(x) x;
		trainTargets = cellfun(trainFxn,trainTargets,'UniformOutput',false);

		% To train classifiers on a set of images/signals
		[classifyStructTraining] = classifySignals(inputImages(trainSessionNo),inputSignals(trainSessionNo),...
			'classifierType','all',...
			'trainingOrClassify','training',...
			'inputMovieList',inputMovie(trainSessionNo),...
			'inputTargets',trainTargets,...
			'readMovieChunks',readMovieChunks,...
			'costStructure',costStructure,...
			'plotTargetVsFeatures',plotTargetVsFeatures,...
			'movieFeatures',movieFeatures,...
			'preComputedFeatures',preComputedFeatures(trainSessionNo),...
			'inputDatasetName',h5DatasetName);

		fileInfoTraining = getFileInfo(inputMoviePaths{movieNo});

		% Classifying all movies, do not include training sessions in classifier
		nMoviesNew = length(classifyOnlySessionNo);
		for movieNoNew = 1:nMoviesNew
			movieNo = classifyOnlySessionNo(movieNoNew);
			fprintf('Classifying %d/%d: %s\n',movieNoNew,nMoviesNew,inputMoviePaths{movieNo})
			% Classify a set of images/signals
			if isempty(inputTargets{movieNo})
				testTargets = {[]};
			else
				testTargets = inputTargets(movieNo);
			end
			[folderClassifier] = classifySignals(inputImages(movieNo),inputSignals(movieNo),...
				'classifierType','all',...
				'trainingOrClassify','classify',...
				'inputMovieList',inputMovie(movieNo),...
				'inputTargets',testTargets,...
				'readMovieChunks',readMovieChunks,...
				'inputStruct',classifyStructTraining,...
				'costStructure',costStructure,...
				'plotConfusion',0,...
				'plotTargetVsFeatures',0,...
				'movieFeatures',movieFeatures,...
				'preComputedFeatures',preComputedFeatures(movieNo),...
				'inputDatasetName',h5DatasetName);

			[pathstr,name,ext] = fileparts(inputMoviePaths{movieNo});
			saveStr = [inputMoviePaths{movieNo} filesep name saveNameClassifierStruct '.mat'];
			classifierPredictions = folderClassifier.classificationsModelDecisions;
			validCELLMax = logical(classifierPredictions(:)>classificationsThreshold);
			validCellMax = logical(classifierPredictions(:)>classificationsThreshold);
			fprintf('Saving to: %s\n',saveStr);
			save(saveStr,'validCELLMax','validCellMax','folderClassifier','-v7.3');
		end
	end
	return;
end

% Create classifier both normal, shuffled, and setting targets to various extreme cases
analysisState = {'normal','shuffle','all_cell','all_nonCell','random'};
nStates = length(analysisState);
for analysisNo = 1:nStates
	analysisStateStr = analysisState{analysisNo};
	display(repmat('=',1,42))
	fprintf('%d/%d: %s\n',analysisNo,nStates,analysisStateStr)
	plotTargetVsFeatures = 0;

	% Different target states for classification
	if strcmp(analysisStateStr,'normal')
		trainTargets = inputTargets(trainSessionNo);
		plotTargetVsFeatures = 1;
		trainFxn = @(trainTargets) trainTargets;
	elseif strcmp(analysisStateStr,'all_cell')
		trainTargets = inputTargets(trainSessionNo);
		% trainTargets = logical(trainTargets*0+1);
		trainFxn = @(trainTargets) logical(trainTargets*0+1);
	elseif strcmp(analysisStateStr,'all_nonCell')
		trainTargets = inputTargets(trainSessionNo);
		% trainTargets = logical(trainTargets*0);
		trainFxn = @(trainTargets) logical(trainTargets*0);
	elseif strcmp(analysisStateStr,'random')
		trainTargets = inputTargets(trainSessionNo);
		% trainTargets = rand(size(trainTargets))>0.5;
		trainFxn = @(trainTargets) rand(size(trainTargets))>0.5;
	elseif strcmp(analysisStateStr,'shuffle')
		trainTargets = inputTargets(trainSessionNo);
		% trainTargets = trainTargets(randperm(length(trainTargets)));
		trainFxn = @(trainTargets) trainTargets(randperm(length(trainTargets)));
	end
	trainTargets = cellfun(trainFxn,trainTargets,'UniformOutput',false);

	if strcmp(analysisStateStr,'normal')|strcmp(analysisStateStr,'shuffle')
		% To train classifiers on a set of images/signals
		[classifyStructTraining] = classifySignals(inputImages(trainSessionNo),inputSignals(trainSessionNo),...
			'classifierType','all',...
			'trainingOrClassify','training',...
			'inputMovieList',inputMovie(trainSessionNo),...
			'inputTargets',trainTargets,...
			'readMovieChunks',readMovieChunks,...
			'costStructure',costStructure,...
			'plotTargetVsFeatures',plotTargetVsFeatures,...
			'movieFeatures',movieFeatures,...
			'preComputedFeatures',preComputedFeatures(trainSessionNo),...
			'inputDatasetName',h5DatasetName);
	end

	fileInfoTraining = getFileInfo(inputMoviePaths{movieNo});
	end2 = toc(startTime);

	% Classifying all movies, do not include training sessions in classifier
	for movieNo = 1:nMovies
		display(repmat('+',1,21))
		fprintf('%d/%d: %s | %d/%d: %s\n',analysisNo,nStates,analysisStateStr,movieNo,nMovies,inputMoviePaths{movieNo})
		% if trainSessionNo==movieNo
		%     disp('SKIP TRAINING SESSION WHEN TESTING')
		%     continue;
		% end

		if ~exist('summaryStats','var')
			summaryStats.TN{1,1} = nan;
			summaryStats.FN{1,1} = nan;
			summaryStats.FP{1,1} = nan;
			summaryStats.TP{1,1} = nan;
			summaryStats.classifierType{1,1} = nan;
			summaryStats.subject{1,1} = nan;
			summaryStats.subjectProtocol{1,1} = nan;
			summaryStats.protocol{1,1} = nan;
			summaryStats.assay{1,1} = nan;
			summaryStats.assayType{1,1} = nan;
			summaryStats.assayNum{1,1} = nan;
			summaryStats.imagingPlane{1,1} = nan;
			summaryStats.trainingState{1,1} = nan;
			summaryStats.trainingSubject{1,1} = nan;
			summaryStats.testSubject{1,1} = nan;
			summaryStats.signalExtractionMethod{1,1} = nan;
		end

		if strcmp(analysisStateStr,'normal')|strcmp(analysisStateStr,'shuffle')
			% Classify a set of images/signals
			[folderClassifier] = classifySignals({inputImages{movieNo}},{inputSignals{movieNo}},...
				'classifierType','all',...
				'trainingOrClassify','classify',...
				'inputMovieList',{inputMovie{movieNo}},...
				'inputTargets',{inputTargets{movieNo}},...
				'readMovieChunks',readMovieChunks,...
				'inputStruct',classifyStructTraining,...
				'costStructure',costStructure,...
				'plotTargetVsFeatures',0,...
				'movieFeatures',movieFeatures,...
				'preComputedFeatures',{preComputedFeatures{movieNo}},...
				'inputDatasetName',h5DatasetName);
		else
			if strcmp(analysisStateStr,'normal')
				testTargets = inputTargets{movieNo};
			elseif strcmp(analysisStateStr,'all_cell')
				testTargets = inputTargets{movieNo};
				testTargets = logical(testTargets*0+1);
			elseif strcmp(analysisStateStr,'all_nonCell')
				testTargets = inputTargets{movieNo};
				testTargets = logical(testTargets*0);
			elseif strcmp(analysisStateStr,'random')
				testTargets = inputTargets{movieNo};
				testTargets = rand(size(testTargets))>0.5;
			elseif strcmp(analysisStateStr,'shuffle')
				testTargets = inputTargets{movieNo};
				testTargets = testTargets(randperm(length(testTargets)));
			end
			folderClassifier.svmGroups = testTargets;
			folderClassifier.nnetGroups = testTargets;
			folderClassifier.glmGroups = testTargets;
			folderClassifier.naiveBayesGroups = testTargets;
			folderClassifier.decisionTreeGroups = testTargets;
			folderClassifier.classifications = testTargets;
			folderClassifier.classificationsModelDecisions = testTargets;
		end

		if any(ismember(trainSessionNo,movieNo))==0&&~isempty(inputTargets{movieNo})
			classifierTypes = {'svm','nnet','glm','naiveBayes','decisionTree','svm_nnet_glm_naiveBayes_decisionTree','consensus_model'};
			% obj.classifierFolderStructs{obj.fileNum}.confusionPct{classifierNo}(1);

			for classifierNo = 1:length(classifierTypes)
				% folderClassifier = obj.classifierFolderStructs{obj.fileNum}.(analysisStateStr);
				switch classifierTypes{classifierNo}
					case 'svm'
						classifierPredictions = folderClassifier.svmGroups;
						classifierPredictions = classifierPredictions+classificationsThreshold;
					case 'nnet'
						classifierPredictions = folderClassifier.nnetGroups;
					case 'glm'
						classifierPredictions = folderClassifier.glmGroups;
					case 'naiveBayes'
						classifierPredictions = folderClassifier.naiveBayesGroups;
					case 'decisionTree'
						classifierPredictions = folderClassifier.decisionTreeGroups;
					case 'svm_nnet_glm_naiveBayes_decisionTree'
						classifierPredictions = folderClassifier.classifications;
					case 'consensus_model'
						classifierPredictions = folderClassifier.classificationsModelDecisions;
					otherwise
						% body
				end

				% perTab = crosstab(logical(inputTargets(:)),logical(classifierPredictions(:)>0.5));
				perTab = confusionmat(logical(inputTargets{movieNo}(:)),logical(classifierPredictions(:)>classificationsThreshold));
				perTab = perTab(:);

				fileInfo = getFileInfo(inputMoviePaths{movieNo});

				summaryStats.TN{end+1,1} = perTab(1);
				summaryStats.FN{end+1,1} = perTab(2);
				summaryStats.FP{end+1,1} = perTab(3);
				summaryStats.TP{end+1,1} = perTab(4);
				summaryStats.classifierType{end+1,1} = classifierTypes{classifierNo};
				summaryStats.subject{end+1,1} = fileInfo.subjectStr;
				summaryStats.subjectProtocol{end+1,1} = [fileInfo.subjectStr '_' fileInfo.protocol];
				summaryStats.protocol{end+1,1} = fileInfo.protocol;
				summaryStats.assay{end+1,1} = fileInfo.assay;
				summaryStats.assayType{end+1,1} = fileInfo.assayType;
				summaryStats.assayNum{end+1,1} = fileInfo.assayNum;
				summaryStats.imagingPlane{end+1,1} = fileInfo.imagingPlane;
				summaryStats.trainingState{end+1,1} = {analysisStateStr};
				summaryStats.trainingSubject{end+1,1} = [fileInfoTraining.subjectStr];
				summaryStats.testSubject{end+1,1} = fileInfo.subjectStr;
				summaryStats.signalExtractionMethod{end+1,1} = {signalExtractionMethod};
			end

			if exist('summaryStats','var')
				savePath = [dataSaveOutputPath filesep runID '_classifySummary.tab'];
				display(['saving data to: ' savePath])
				writetable(struct2table(summaryStats),savePath,'FileType','text','Delimiter','\t');
			end
		end

		% Save model classifications along with full classifier
		if strcmp(analysisStateStr,'normal')&&saveClassiferOutput==1
			[pathstr,name,ext] = fileparts(inputMoviePaths{movieNo});
			saveStr = [inputMoviePaths{movieNo} filesep name saveNameClassifierStruct '.mat'];
			classifierPredictions = folderClassifier.classificationsModelDecisions;
			validCELLMax = logical(classifierPredictions(:)>classificationsThreshold);
			validCellMax = logical(classifierPredictions(:)>classificationsThreshold);
			fprintf('Saving to: %s\n',saveStr);
			save(saveStr,'validCELLMax','validCellMax','folderClassifier','-v7.3');
		end
	end
end

end3 = toc(startTime);

if exist('end0','var'); disp(['end0: ' num2str(end0)]); end
if exist('end1','var'); disp(['end1: ' num2str(end1)]); end
if exist('end2','var'); disp(['end2: ' num2str(end2)]); end
if exist('end3','var'); disp(['end3: ' num2str(end3)]); end

% profile viewer

toc