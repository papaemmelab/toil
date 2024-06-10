from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

from .listener import IListenerExcHandler, Listener, UserListener
from .notificationmgr import INotificationHandler
from .topicmgr import TopicManager, TreeConfig

TopicFilter = Callable[[str], bool]
ListenerFilter = Callable[[Listener], bool]

class Publisher:
    def __init__(self, treeConfig: TreeConfig = ...) -> None: ...
    def getTopicMgr(self) -> TopicManager: ...
    def getListenerExcHandler(self) -> IListenerExcHandler: ...
    def setListenerExcHandler(self, handler: IListenerExcHandler) -> None: ...
    def addNotificationHandler(self, handler: INotificationHandler) -> None: ...
    def clearNotificationHandlers(self) -> None: ...
    def setNotificationFlags(self, **kwargs: Mapping[str, Optional[bool]]) -> None: ...
    def getNotificationFlags(self) -> Mapping[str, bool]: ...
    def setTopicUnspecifiedFatal(
        self, newVal: bool = ..., checkExisting: bool = ...
    ) -> bool: ...
    def subscribe(
        self, listener: UserListener, topicName: str, **curriedArgs: Any
    ) -> Tuple[Listener, bool]: ...
    def unsubscribe(self, listener: UserListener, topicName: str) -> Listener: ...
    def unsubAll(
        self,
        topicName: str = ...,
        listenerFilter: ListenerFilter = ...,
        topicFilter: Union[str, TopicFilter] = ...,
    ) -> List[Listener]: ...
    def sendMessage(self, topicName: str, **msgData: Any) -> None: ...