use crate::{bulletin_board::BulletinBoardId, transaction::TxData, wallet::WalletId, TimeStep};

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct PayjoinProposal {
    pub(crate) tx: TxData,
    pub(crate) valid_till: TimeStep,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum MessageType {
    /// Initiate a cospend with the receiver of payment
    InitiatePayjoin(BulletinBoardId),
}

define_entity!(
    Message,
    {
        pub(crate) id: MessageId,
        pub(crate) message: MessageType,
        pub(crate) from: WalletId,
        // None if meant as a broadcast message
        pub(crate) to: WalletId,
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
