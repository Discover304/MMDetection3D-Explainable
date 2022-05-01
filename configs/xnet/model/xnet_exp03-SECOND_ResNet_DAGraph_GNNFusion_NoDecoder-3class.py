_base_ = [
    './xnet_exp01_3-SECOND_ResNet_DNNGraph_GNNFusion_NoDecoder-3class.py'
    ]

model = dict(
    get_graph=dict(
        _delete_=True,
        type='GetGraphDAG',
        in_channel=64+64)
)
