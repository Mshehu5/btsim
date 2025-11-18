use crate::{cospend::CospendId, wallet::WalletId};

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct InitiateCospend {
    pub(crate) cospend_id: CospendId,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum MessageType {
    /// Initiate a cospend with the receiver of payment
    RegisterCospend(InitiateCospend),
}

define_entity!(
    Message,
    {
        pub(crate) id: MessageId,
        pub(crate) message: MessageType,
        pub(crate) from: WalletId,
        // None if meant as a broadcast message
        pub(crate) to: Option<WalletId>,
    },
    {
    }
);
define_entity_handle_mut!(Message);

impl<'a> MessageHandle<'a> {
    pub(crate) fn data(&self) -> &'a MessageData {
        &self.sim.messages[self.id.0]
    }
}

impl<'a> MessageHandleMut<'a> {
    pub(crate) fn post(&mut self, message: MessageData) {
        self.sim.messages.push(message);
    }
}
