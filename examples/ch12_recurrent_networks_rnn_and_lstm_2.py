# Chapter 12 — 12.3 Recurrent Networks: RNN and LSTM
self.lstm = nn.LSTM(
    input_size=input_dim,
    hidden_size=hidden_dim,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=True,       # forward AND backward
)
# the output width doubles: forward || backward
self.classifier = nn.Linear(hidden_dim * 2, num_classes)
