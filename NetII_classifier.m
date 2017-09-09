X=test_outputs(:,:);
train_X=X(:,(1:42))
test_X=X(:,(43:46))

Y=targetsoneHot;
train_Y=Y(:,1:42)
test_Y=Y(:,43:46)

Y_label=classifier_targets;
train_Y_label=Y_label(1:42)
test_Y_label=Y_label(43:46)

trainFcn = 'trainlm';

hiddenLayerSize = 3;

net = patternnet(hiddenLayerSize,trainFcn);

net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

%net.performFcn = 'crossentropy';
net.performFcn = 'mse';  % Mean Squared Error

net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,train_X,train_Y);

% Validating the Network
train_out = net(train_X);
class_output = vec2ind(train_out)
e = gsubtract(class_output,train_Y_label)
performance = perform(net,train_Y_label,class_output);

%Testing on test data
final_output=net(test_X)
class_final_output=vec2ind(final_output)
testing_error = gsubtract(class_final_output,test_Y_label);
performance = perform(net,train_Y_label,class_output);