warning('off','all')
addpath('C:\Users\DavideDP\AnacondaProjects\Project\RankFusion\core\io')
addpath('C:\Users\DavideDP\AnacondaProjects\Project\RankFusion\core\measure')
addpath('C:\Users\DavideDP\AnacondaProjects\Project\RankFusion\core\util')
addpath('C:\Users\DavideDP\AnacondaProjects\Project\RankFusion')

%path_to_pool, path_run,name_run

%import Pool
%path_to_pool='C:\Users\DavideDP\AnacondaProjects\Project\terrier-core-4.2\share\TIPSTER\pool\qrel.txt'
%path_to_run='C:\Users\DavideDP\AnacondaProjects\Project\RankFusion\input\BM25b0.75_0.res'
%name_run='BM25b0.75_0.res'

[pool, report] = importPoolFromFileTRECFormat('FileName',path_to_pool, 'Identifier', 'p1', 'RelevanceGrades', 0:1, 'RelevanceDegrees', {'NotRelevant', 'Relevant'});

    
%import Run
[runSet, report2] = importRunFromFileTRECFormat('FileName', path_to_run, 'Identifier', 'p2', 'Delimiter', 'Space');

%assess
[assessedRunSet, poolStats, runSetStats, inputParams] = assess(pool, runSet);
    
% averagePrecision
measuredRunSet =  averagePrecision(pool, runSet);

sum=0
for k = 351:400
   sum=sum+measuredRunSet{int2str(k),1}
end
sum=sum/50


